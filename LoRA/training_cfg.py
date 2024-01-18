from dataclasses import dataclass
from omegaconf import OmegaConf


@dataclass
class TrainingConfig:
    log_dir: str
    output_dir: str
    data_dir: str
    ckpt_name: str
    rank: int
    gradient_accumulation_steps: int = 1
    mixed_precision: str = None
    seed: int = None
    pretrained_model_name_or_path: str = 'ckpt/v1-5'
    enable_xformers_memory_efficient_attention: bool = True

    # AdamW
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    resolution: int = 512
    n_epochs: int = 200
    checkpointing_steps: int = 500
    train_batch_size: int = 1
    dataloader_num_workers: int = 1

    lr_scheduler_name: str = 'constant'

    resume_from_checkpoint: bool = False
    noise_offset: float = 0.1
    max_grad_norm: float = 1.0


def load_training_config(config_path: str) -> TrainingConfig:
    data_dict = OmegaConf.load(config_path)
    return TrainingConfig(**data_dict)


if __name__ == '__main__':
    config = load_training_config('config/train/mountain.json')
    print(config)
