import inspect
import os
from dataclasses import dataclass

from diffusers import AutoencoderKL, DDPMScheduler, DiffusionPipeline, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import check_min_version, convert_state_dict_to_diffusers, is_wandb_available
import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict

from trainer import Trainer


@dataclass
class LoraTrainingConfig:
    # Diffuion Models
    pretrained_model_name_or_path: str
    revision: str = None
    variant: str = None
    rank: int = 4
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100

    max_grad_norm = 0.1

    # Validation
    valid_seed = 0
    valid_batch_size: int = 1

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500


def log_validation(
    pipeline,
    seed,
    num_validation_images,
    accelerator,
    epoch,
    is_final_validation=False,
):
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if seed is not None:
        generator = generator.manual_seed(seed)
    images = []

    autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(num_validation_images):
            images.append(pipeline("", num_inference_steps=30,
                          generator=generator).images[0])

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(
                phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            import wandb
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {''}") for i, image in enumerate(images)
                    ]
                }
            )
    return images


class LoraTrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: LoraTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        cfg = self.cfg
        # Load scheduler, tokenizer and models.
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="scheduler")
        self.tokenizer = CLIPTokenizer.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="tokenizer", revision=cfg.revision
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="text_encoder", revision=cfg.revision
        )
        self.vae = AutoencoderKL.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="vae", revision=cfg.revision, variant=cfg.variant
        )
        self.unet = UNet2DConditionModel.from_pretrained(
            cfg.pretrained_model_name_or_path, subfolder="unet", revision=cfg.revision, variant=cfg.variant
        )
        # freeze parameters of models to save more memory
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)

        for param in self.unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )

        self.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.vae.to(self.accelerator.device, dtype=self.weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        self.unet.add_adapter(unet_lora_config)
        if self.accelerator.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(self.unet, dtype=torch.float32)

        if enable_xformer:
            self.unet.enable_xformers_memory_efficient_attention()

        self.lora_layers = filter(
            lambda p: p.requires_grad, self.unet.parameters())

        if gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()

        self.empty_ids = self.tokenizer(
            '', max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids

    def init_optimizers(self, train_batch_size):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * self.cfg.gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )
        self.optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=self.cfg.learning_rate,
            betas=(self.cfg.adam_beta1, self.cfg.adam_beta2),
            weight_decay=self.cfg.adam_weight_decay,
            eps=self.cfg.adam_epsilon,
        )

    def init_lr_schedulers(self, gradient_accumulation_steps, num_epochs):
        self.lr_scheduler = get_scheduler(
            self.cfg.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.cfg.lr_warmup_steps *
            gradient_accumulation_steps,
            num_training_steps=(len(self.train_dataloader)
                                * num_epochs)
        )

    def prepare_modules(self):
        self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.unet, self.optimizer, self.train_dataloader, self.lr_scheduler
        )

    def models_to_train(self):
        self.unet.train()

    def training_step(self, global_step, batch) -> dict:
        train_loss = 0.0
        with self.accelerator.accumulate(self.unet):
            # Convert images to latent space
            latents = self.vae.encode(batch["input"].to(
                dtype=self.weight_dtype)).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            input_ids = self.empty_ids.repeat(latents.shape[0], 1)
            input_ids = input_ids.to(latents.device)

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)

            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = self.noise_scheduler.add_noise(
                latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = self.text_encoder(
                input_ids, return_dict=False)[0]

            # Get the target for loss depending on the prediction type
            if self.cfg.prediction_type is not None:
                # set prediction_type of scheduler if defined
                self.noise_scheduler.register_to_config(
                    prediction_type=self.cfg.prediction_type)

            if self.noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self.noise_scheduler.config.prediction_type == "v_prediction":
                target = self.noise_scheduler.get_velocity(
                    latents, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {self.noise_scheduler.config.prediction_type}")

            # Predict the noise residual and compute loss
            model_pred = self.unet(noisy_latents, timesteps,
                                   encoder_hidden_states, return_dict=False)[0]

            loss = F.mse_loss(model_pred.float(),
                              target.float(), reduction="mean")

            train_batch_size = latents.shape[0]

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = self.accelerator.gather(
                loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / self.accelerator.gradient_accumulation_steps

            # Backpropagate
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                params_to_clip = self.lora_layers
                self.accelerator.clip_grad_norm_(
                    params_to_clip, self.cfg.max_grad_norm)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            logs = {"train_loss": train_loss}

        return logs

    def validate(self, epoch, global_step):
        pipeline = DiffusionPipeline.from_pretrained(
            self.cfg.pretrained_model_name_or_path,
            unet=self.accelerator.unwrap_model(self.unet),
            revision=self.cfg.revision,
            variant=self.cfg.variant,
            torch_dtype=self.weight_dtype,
        )
        log_validation(
            pipeline, self.cfg.valid_seed, self.cfg.valid_batch_size, self.accelerator, epoch)

        del pipeline
        torch.cuda.empty_cache()

    def save_pipeline(self, output_dir):
        self.unet = self.unet.to(torch.float32)

        unwrapped_unet = self.accelerator.unwrap_model(self.unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )

        StableDiffusionPipeline.save_lora_weights(
            save_directory=output_dir,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            for i, model in enumerate(models):
                unwrapped_unet = self.accelerator.unwrap_model(model)
                unet_lora_state_dict = convert_state_dict_to_diffusers(
                    get_peft_model_state_dict(unwrapped_unet)
                )

                StableDiffusionPipeline.save_lora_weights(
                    save_directory=output_dir,
                    unet_lora_layers=unet_lora_state_dict,
                    safe_serialization=True,
                )

    def load_model_hook(self, models, input_dir):
        pass
