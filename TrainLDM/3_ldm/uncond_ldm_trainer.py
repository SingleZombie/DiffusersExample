import inspect
import os

from dataclasses import dataclass
from diffusers import DDPMPipeline, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version
import torch
import torch.nn.functional as F
from torchmetrics.image.fid import FrechetInceptionDistance
import line_profiler

from trainer import Trainer
from dataset import DataKey
from my_unet import MyUnet
from ddpm_trainer import DDPMTrainer
from uncond_ldm_pipeline import UncondLDMPipeline


@dataclass
class UncondLDMTrainingConfig:
    # Diffuion Models
    model_config: str
    scheduler_config: str
    vae_dir: str
    num_inference_steps: int = 100

    # Validation
    valid_batch_size: int = 1
    valid_loops: int = 100
    valid_fid: bool = False

    # EMA
    use_ema: bool = False
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4

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


class UncondLDMTrainer(DDPMTrainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: UncondLDMTrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):
        config = MyUnet.load_config(self.cfg.model_config)
        self.model = MyUnet.from_config(config)

        self.vae = AutoencoderKL.from_pretrained(self.cfg.vae_dir)

        self.noise_scheduler = DDPMScheduler.from_config(
            DDPMScheduler.load_config(self.cfg.scheduler_config)
        )

        # Create EMA for the model.
        if self.cfg.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=self.cfg.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.cfg.ema_inv_gamma,
                power=self.cfg.ema_power,
                model_cls=MyUnet,
                model_config=self.model.config,
            )

        if enable_xformer:
            self.model.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

    def prepare_modules(self):
        super().prepare_modules()

        self.vae.to(self.accelerator.device)

    @line_profiler.profile
    def training_step(self, global_step, batch) -> dict:
        weight_dtype = self.weight_dtype
        clean_images = batch[DataKey.IMAGE].to(weight_dtype)

        bsz = clean_images.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()

        with torch.no_grad():
            latents = self.vae.encode(clean_images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = latents.to(dtype=self.weight_dtype)

        # Sample noise that we'll add to the images
        noise = torch.randn(latents.shape,
                            dtype=weight_dtype, device=latents.device)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(
            latents, noise, timesteps)

        with self.accelerator.accumulate(self.model):
            # Predict the noise residual
            model_output = self.model(noisy_images, timesteps).sample

            if self.cfg.prediction_type == "epsilon":
                # this could have different weights!
                loss = F.mse_loss(model_output.float(), noise.float())
            else:
                raise ValueError(
                    f"Unsupported prediction type: {self.cfg.prediction_type}")

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

        if self.accelerator.sync_gradients:
            if self.cfg.use_ema:
                self.ema_model.step(self.model.parameters())

        logs = {"loss": loss.detach().item()}
        if self.cfg.use_ema:
            logs["ema_decay"] = self.ema_model.cur_decay_value

        return logs

    def validate(self, epoch, global_step):
        unet = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = UncondLDMPipeline(
            vae=self.vae,
            unet=unet,
            scheduler=self.noise_scheduler,
        ).to(unet.device)
        pipeline.set_progress_bar_config(disable=True)

        # run pipeline in inference (sample random noise and denoise)
        if self.cfg.valid_fid:
            for _ in range(self.cfg.valid_loops):
                images = pipeline(
                    batch_size=self.cfg.valid_batch_size,
                    num_inference_steps=self.cfg.num_inference_steps,
                    output_type="np",
                ).images
                image_tensor = torch.from_numpy(
                    images).permute(0, 3, 1, 2).to(self.accelerator.device)
                self.fid.update(image_tensor, False)

        generator = torch.Generator(
            device=pipeline.device).manual_seed(0)
        images = pipeline(
            generator=generator,
            batch_size=self.cfg.valid_batch_size,
            num_inference_steps=self.cfg.num_inference_steps,
            output_type="pt",
        ).images

        if self.cfg.use_ema:
            self.ema_model.restore(unet.parameters())

        # denormalize the images and save to tensorboard

        if self.cfg.valid_fid:
            msg_dict = {'fid': self.fid.compute()}
        else:
            msg_dict = {}
        self.accelerator.log(msg_dict, step=global_step)

        if self.logger == "tensorboard":
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = self.accelerator.get_tracker(
                    "tensorboard", unwrap=True)
            else:
                tracker = self.accelerator.get_tracker("tensorboard")
            tracker.add_images(
                "test_samples", images.transpose(0, 3, 1, 2), epoch)
        elif self.logger == "wandb":
            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            import wandb
            self.accelerator.get_tracker("wandb").log(
                {"test_samples": [wandb.Image(
                    img) for img in images], "epoch": epoch},
                step=global_step,
            )

    def save_pipeline(self, output_dir):
        unet = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(unet.parameters())
            self.ema_model.copy_to(unet.parameters())

        pipeline = UncondLDMPipeline(
            vae=self.vae,
            unet=unet,
            scheduler=self.noise_scheduler,
        )

        pipeline.save_pretrained(output_dir)

        if self.cfg.use_ema:
            self.ema_model.restore(unet.parameters())
