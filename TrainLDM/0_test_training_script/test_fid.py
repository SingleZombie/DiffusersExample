from dataset import create_dataset, DataKey
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import sys
from training_cfg import BaseTrainingConfig, load_training_config
from ddpm_trainer import DDPMTrainingConfig

cfg_path = sys.argv[1]
cfgs = load_training_config(cfg_path)
cfg: BaseTrainingConfig = cfgs.pop('base')
trainer_type = next(iter(cfgs))
trainer_cfg: DDPMTrainingConfig = cfgs[trainer_type]


valid_dataset = create_dataset(cfg.resolution,
                               cfg.train_dataset_name,
                               cfg.train_dataset_config_name,
                               cfg.train_data_dir,
                               cfg.train_data_files,
                               cache_dir=cfg.cache_dir,
                               center_crop=cfg.center_crop,
                               random_flip=cfg.random_flip)

valid_dataloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=cfg.train_batch_size, shuffle=False,
    num_workers=cfg.dataloader_num_workers
)

device = 'cuda'

fid = FrechetInceptionDistance(2048, normalize=True).to(device)

for i, batch in enumerate(valid_dataloader):
    fid.update(batch[DataKey.IMAGE].to(device), real=True)
    if i > 1000:
        break

model_path = cfg.output_dir
unet = UNet2DModel.from_pretrained(model_path)
noise_scheduler = DDPMScheduler(
    num_train_timesteps=trainer_cfg.ddpm_num_steps,
    beta_schedule=trainer_cfg.ddpm_beta_schedule,
    prediction_type=trainer_cfg.prediction_type,
)
pipeline = DDPMPipeline(
    unet=unet,
    scheduler=noise_scheduler,
)
pipeline.set_progress_bar_config(disable=True)

# run pipeline in inference (sample random noise and denoise)
for _ in range(100):
    images = pipeline(
        batch_size=trainer_cfg.valid_batch_size,
        num_inference_steps=trainer_cfg.ddpm_num_inference_steps,
        output_type="np",
    ).images
    image_tensor = torch.from_numpy(
        images).permute(0, 3, 1, 2).to(device)
    fid.update(image_tensor, False)

print('FID:', fid.compute().item())
