import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import datetime
from pathlib import Path

import accelerate
import datasets
import torch

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers.utils import check_min_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from training_cfg_1 import BaseTrainingConfig, load_training_config
from trainer import Trainer, create_trainer


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('cfg', type=str)
    args = parser.parse_args()

    cfgs = load_training_config(args.cfg)
    cfg: BaseTrainingConfig = cfgs.pop('base')
    trainer_type = next(iter(cfgs))
    trainer_cfg_dict = cfgs[trainer_type]

    logging_dir = os.path.join(cfg.output_dir, cfg.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=cfg.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.logger,
        project_config=accelerator_project_config
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        cfg.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        cfg.mixed_precision = accelerator.mixed_precision

    trainer: Trainer = create_trainer(
        trainer_type, weight_dtype, accelerator, cfg.logger, trainer_cfg_dict)

    if cfg.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Make sure to install tensorboard if you want to use it for logging during training.")

    elif cfg.logger == "wandb":
        if not is_wandb_available():
            raise ImportError(
                "Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        accelerator.register_save_state_pre_hook(trainer.save_model_hook)
        accelerator.register_load_state_pre_hook(trainer.load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            os.makedirs(cfg.output_dir, exist_ok=True)

        if cfg.push_to_hub:
            repo_id = create_repo(
                repo_id=cfg.hub_model_id or Path(cfg.output_dir).name, exist_ok=True, token=cfg.hub_token
            ).repo_id

    # Initialize the model
    enable_xformers = False
    if cfg.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            enable_xformers = True

        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly")

    trainer.init_modules(enable_xformers, cfg.gradient_checkpointing)

    # Initialize the optimizer
    trainer.init_optimizers(cfg.train_batch_size)

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).

    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    if cfg.dataset_name is not None:
        dataset = load_dataset(
            cfg.dataset_name,
            cfg.dataset_config_name,
            cache_dir=cfg.cache_dir,
            split="train",
        )
    else:
        dataset = load_dataset(
            "imagefolder", data_dir=cfg.train_data_dir, cache_dir=cfg.cache_dir, split="train")
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    # Preprocessing the datasets and DataLoaders creation.
    augmentations = transforms.Compose(
        [
            transforms.Resize(
                cfg.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                cfg.resolution) if cfg.center_crop else transforms.RandomCrop(cfg.resolution),
            transforms.RandomHorizontalFlip() if cfg.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        images = [augmentations(image.convert("RGB"))
                  for image in examples["image"]]
        return {"input": images}

    logger.info(f"Dataset size: {len(dataset)}")

    dataset.set_transform(transform_images)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.train_batch_size, shuffle=True, num_workers=cfg.dataloader_num_workers
    )

    trainer.set_dataset(dataset, train_dataloader)

    # Initialize the learning rate scheduler
    trainer.init_lr_schedulers(cfg.gradient_accumulation_steps, cfg.num_epochs)

    # Prepare everything with our `accelerator`.
    trainer.prepare_modules()
    train_dataloader = trainer.train_dataloader

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        now = datetime.now()
        formatted_now = now.strftime('%Y%m%d%H%M%S')
        accelerator.init_trackers(
            formatted_now, config=vars(args))

    total_batch_size = cfg.train_batch_size * \
        accelerator.num_processes * cfg.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.gradient_accumulation_steps)
    max_train_steps = cfg.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_epochs}")
    logger.info(
        f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            path = os.path.basename(cfg.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(cfg.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{cfg.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            cfg.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(cfg.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * cfg.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * cfg.gradient_accumulation_steps)

    # Train!
    for epoch in range(first_epoch, cfg.num_epochs):
        trainer.models_to_train()
        progress_bar = tqdm(total=num_update_steps_per_epoch,
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if cfg.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % cfg.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            logs = trainer.training_step(global_step, batch)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % cfg.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if cfg.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(cfg.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= cfg.checkpoints_total_limit:
                                num_to_remove = len(
                                    checkpoints) - cfg.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        cfg.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(
                            cfg.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % cfg.valid_epochs == 0 or epoch == cfg.num_epochs - 1:
                trainer.validate(epoch, global_step)

            if epoch % cfg.save_model_epochs == 0 or epoch == cfg.num_epochs - 1:
                trainer.save_pipeline(cfg.output_dir)

                if cfg.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=cfg.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )

    accelerator.end_training()


if __name__ == "__main__":
    main()
