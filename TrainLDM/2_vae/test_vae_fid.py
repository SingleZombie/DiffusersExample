from dataset import create_dataset, DataKey
from diffusers import DDPMPipeline, DDPMScheduler, DDIMPipeline
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
import sys
from training_cfg import BaseTrainingConfig, load_training_config
from ddpm_trainer import DDPMTrainingConfig
from my_unet import MyUnet
from tqdm import tqdm
import os
import line_profiler
from torchvision.utils import save_image

n_training_data_batch = None  # None for the entire dataset
n_sample_data_batch = 50
valid_batch_size = 20


@line_profiler.profile
def main():
    cfg_path = sys.argv[1]
    cfgs = load_training_config(cfg_path)
    cfg: BaseTrainingConfig = cfgs.pop('base')
    trainer_type = next(iter(cfgs))
    trainer_cfg: DDPMTrainingConfig = cfgs[trainer_type]

    if len(sys.argv) >= 3:
        fid_state_path = sys.argv[2]
    else:
        fid_state_path = None

    valid_dataset = create_dataset(cfg.resolution,
                                   cfg.train_dataset_name,
                                   cfg.train_dataset_config_name,
                                   cfg.train_data_dir,
                                   cfg.train_data_files,
                                   cache_dir=cfg.cache_dir,
                                   center_crop=False,
                                   random_flip=False)

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=cfg.train_batch_size, shuffle=False,
        num_workers=cfg.dataloader_num_workers
    )

    device = 'cuda'

    fid = FrechetInceptionDistance(normalize=True).to(device)
    fid.persistent(True)

    # counter = 0
    # for i, batch in enumerate(valid_dataloader):
    #     batch_img = batch[DataKey.IMAGE].to(device)
    #     fid.update(batch_img, real=True)
    #     for img in batch_img:
    #         save_image(
    #             img, f'/home/yfzhou/repo/DiffusersExample/data/ffhq_32/{counter}.png')
    #         counter += 1
    #     break

    if fid_state_path is None or not os.path.exists(fid_state_path):
        for i, batch in enumerate(valid_dataloader):
            if n_training_data_batch is not None and i >= n_training_data_batch:
                break
            batch_img = batch[DataKey.IMAGE].to(device)
            batch_img = (batch_img + 1) / 2
            fid.update(batch_img, real=True)
        if fid_state_path is not None:
            torch.save(fid.state_dict(), fid_state_path)
    else:
        fid.load_state_dict(torch.load(fid_state_path))

    model_path = cfg.output_dir
    unet = MyUnet.from_pretrained(f'{model_path}/unet')
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=trainer_cfg.ddpm_num_steps,
        beta_schedule=trainer_cfg.ddpm_beta_schedule,
        prediction_type=trainer_cfg.prediction_type,
    )
    pipeline = DDIMPipeline(
        unet=unet,
        scheduler=noise_scheduler,
    ).to(device)
    pipeline.set_progress_bar_config(disable=True)

    # run pipeline in inference (sample random noise and denoise)
    for _ in tqdm(range(n_sample_data_batch)):
        images = pipeline(
            batch_size=valid_batch_size,
            num_inference_steps=20,
            output_type="np",
            eta=1
        ).images
        image_tensor = torch.from_numpy(
            images).permute(0, 3, 1, 2).to(device)

        fid.update(image_tensor, False)

    print('FID:', fid.compute().item())


if __name__ == '__main__':
    main()
