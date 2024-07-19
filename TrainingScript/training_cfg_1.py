from typing import Dict
from dataclasses import dataclass
from omegaconf import OmegaConf

from ddpm_trainer import DDPMTrainingConfig
from sd_lora_trainer import LoraTrainingConfig


@dataclass
class BaseTrainingConfig:
    # Dir
    logging_dir: str
    output_dir: str

    # Logger and checkpoint
    logger: str = 'tensorboard'
    checkpointing_steps: int = 500
    checkpoints_total_limit: int = 20
    valid_epochs: int = 100
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None

    # Training
    seed: int = None
    num_epochs: int = 200
    train_batch_size: int = 1
    dataloader_num_workers: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    enable_xformers_memory_efficient_attention: bool = True
    gradient_checkpointing: bool = False

    # Dataset
    dataset_name: str = None
    dataset_config_name: str = None
    train_data_dir: str = None
    cache_dir: str = None
    resolution: int = 512
    center_crop: bool = True
    random_flip: bool = False

    # Hub
    push_to_hub: bool = False
    hub_model_id: str = ''


__TYPE_CLS_DICT = {
    'base': BaseTrainingConfig,
    'ddpm': DDPMTrainingConfig,
    'lora': LoraTrainingConfig
}


def load_training_config(config_path: str) -> Dict[str, BaseTrainingConfig]:
    data_dict = OmegaConf.load(config_path)

    # The config must have a "base" key
    base_cfg_dict = data_dict.pop('base')

    # The config must have one another model config
    assert len(data_dict) == 1
    model_key = next(iter(data_dict))
    model_cfg_dict = data_dict[model_key]
    model_cfg_cls = __TYPE_CLS_DICT[model_key]

    return {'base': BaseTrainingConfig(**base_cfg_dict),
            model_key: model_cfg_cls(**model_cfg_dict)}
