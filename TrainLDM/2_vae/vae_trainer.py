import os

from dataclasses import dataclass
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_accelerate_version
import torch
import torch.nn.functional as F
import lpips
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from trainer import Trainer
from dataset import DataKey


@dataclass
class VAETrainingConfig:
    # Model path
    model_config: str

    # Loss weights
    mse_weight: float = 1.0
    kl_weight: float = 1e-4

    # Validation
    valid_loops: int = 8

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


class TrainingVAE(torch.nn.Module):
    def __init__(self, vae: AutoencoderKL) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, x: torch.Tensor, is_encoder: bool):
        if is_encoder:
            return self.vae.encode(x)
        else:  # is decoder
            return self.vae.decode(x)


class VAETrainer(Trainer):
    def __init__(self, weight_dtype, accelerator, logger, cfg: VAETrainingConfig):
        super().__init__(weight_dtype, accelerator, logger, cfg)

    def init_modules(self,
                     enable_xformer=False,
                     gradient_checkpointing=False):

        model_config = AutoencoderKL.load_config(self.cfg.model_config)
        vae = AutoencoderKL.from_config(model_config)
        self.model = TrainingVAE(vae)

        # Create EMA for the model.
        if self.cfg.use_ema:
            self.ema_model = EMAModel(
                self.model.parameters(),
                decay=self.cfg.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=self.cfg.ema_inv_gamma,
                power=self.cfg.ema_power,
                model_cls=AutoencoderKL,
                model_config=self.model.config,
            )

        if enable_xformer:
            self.model.vae.enable_xformers_memory_efficient_attention()

        if gradient_checkpointing:
            self.model.enable_gradient_checkpointing()

        self.perceptual_loss_fn = lpips.LPIPS(net='vgg')

    def init_optimizers(self, train_batch_size, gradient_accumulation_steps=1):
        if self.cfg.scale_lr:
            self.cfg.learning_rate = (
                self.cfg.learning_rate * gradient_accumulation_steps *
                train_batch_size * self.accelerator.num_processes
            )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
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
        self.model, self.optimizer, self.train_dataloader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.lr_scheduler
        )
        if self.cfg.use_ema:
            self.ema_model.to(self.accelerator.device)
        self.perceptual_loss_fn.to(self.accelerator.device)

    def models_to_train(self):
        self.model.train()

    def training_step(self, global_step, batch) -> dict:
        weight_dtype = self.weight_dtype
        input_batch = batch[DataKey.IMAGE].to(weight_dtype)

        with self.accelerator.accumulate(self.model):
            batchs_size = input_batch.shape[0]
            latent_dist = self.model(input_batch, True).latent_dist
            latents = latent_dist.sample()
            recon_input_batch = self.model(latents, False).sample

            mse_loss = F.mse_loss(input_batch.float(),
                                  recon_input_batch.float(),
                                  reduction="mean")

            kl_loss = torch.sum(latent_dist.kl()) / batchs_size
            loss = self.cfg.mse_weight * mse_loss + \
                self.cfg.kl_weight * kl_loss

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
        tmp_vae: TrainingVAE = self.accelerator.unwrap_model(self.model)

        batch_size = next(iter(self.train_dataloader))[DataKey.IMAGE].shape[0]
        n_samples = self.cfg.valid_loops * batch_size
        tot_mse_loss = 0
        tot_perceptual_loss = 0
        psnr_eval = PeakSignalNoiseRatio().to(self.accelerator.device)
        ssim_eval = StructuralSimilarityIndexMeasure().to(self.accelerator.device)
        for i, x in enumerate(self.train_dataloader):
            if i >= self.cfg.valid_loops:
                break
            x = x[DataKey.IMAGE]

            latent_dist = tmp_vae(x, True).latent_dist
            latents = latent_dist.sample()
            recon_x = tmp_vae(latents, False).sample
            mse_loss = (F.mse_loss(x, recon_x) * batch_size).item()
            perceptual_loss = (self.perceptual_loss_fn(
                x, recon_x).mean() * batch_size).item()
            tot_mse_loss += mse_loss
            tot_perceptual_loss += perceptual_loss

            psnr_eval.update(x, recon_x)
            ssim_eval.update(x, recon_x)

        tot_mse_loss /= n_samples
        tot_perceptual_loss /= n_samples
        psnr = psnr_eval.compute()
        ssim = ssim_eval.compute()
        msg_dict = {'mse': tot_mse_loss, 'perceptual': tot_perceptual_loss,
                    'psnr': psnr, 'ssim': ssim}

        self.accelerator.log(msg_dict, step=global_step)

        # visualization
        input_images = [self.dataset[i][DataKey.IMAGE].unsqueeze(0)
                        for i in range(batch_size)]
        input_images = torch.cat(input_images).to(self.accelerator.device)
        latents = tmp_vae(input_images, True).latent_dist.sample()
        reconstruct_images = tmp_vae(latents, False).sample
        images = torch.cat([input_images, reconstruct_images], 3)
        images = (images + 1) / 2

        if self.logger == "tensorboard":
            if is_accelerate_version(">=", "0.17.0.dev0"):
                tracker = self.accelerator.get_tracker(
                    "tensorboard", unwrap=True)
            else:
                tracker = self.accelerator.get_tracker("tensorboard")
            tracker.add_images(
                "test_samples", images, epoch)
        elif self.logger == "wandb":
            # Upcoming `log_images` helper coming in https://github.com/huggingface/accelerate/pull/962/files
            import wandb
            self.accelerator.get_tracker("wandb").log(
                {"test_samples": [wandb.Image(
                    img) for img in images], "epoch": epoch},
                step=global_step,
            )

    def save_pipeline(self, output_dir):
        vae_wrapper: TrainingVAE = self.accelerator.unwrap_model(self.model)

        if self.cfg.use_ema:
            self.ema_model.store(vae_wrapper.parameters())
            self.ema_model.copy_to(vae_wrapper.parameters())

        vae_wrapper.vae.save_pretrained(output_dir)

        if self.cfg.use_ema:
            self.ema_model.restore(vae_wrapper.parameters())

    def save_model_hook(self, models, weights, output_dir):
        if self.accelerator.is_main_process:
            if self.cfg.use_ema:
                self.ema_model.save_pretrained(
                    os.path.join(output_dir, "vae_ema"))

            for i, model in enumerate(models):
                model.vae.save_pretrained(os.path.join(output_dir, "vae"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

    def load_model_hook(self, models, input_dir):
        if self.cfg.use_ema:
            load_model = EMAModel.from_pretrained(
                os.path.join(input_dir, "vae_ema"), AutoencoderKL)
            self.ema_model.load_state_dict(load_model.state_dict())
            self.ema_model.to(self.accelerator.device)
            del load_model

        for i in range(len(models)):
            # pop models so that they are not loaded again
            model = models.pop()

            # load diffusers style into model
            load_model = AutoencoderKL.from_pretrained(
                input_dir, subfolder="unet")
            model.register_to_config(**load_model.config)

            model.vae.load_state_dict(load_model.state_dict())
            del load_model
