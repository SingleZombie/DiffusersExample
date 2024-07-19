from dataclasses import dataclass


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
    valid_batch_size: int = 1
    save_model_epochs: int = 100
    resume_from_checkpoint: str = None

    # Diffuion Models
    model_config: str = None
    ddpm_num_steps: int = 1000
    ddpm_beta_schedule: str = 'linear'
    prediction_type: str = 'epsilon'
    ddpm_num_inference_steps: int = 100

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
    center_crop: bool = False
    random_flip: bool = False

    # LR Scheduler
    lr_scheduler: str = 'constant'
    lr_warmup_steps: int = 500

    # AdamW
    scale_lr = False
    learning_rate: float = 1e-4
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_weight_decay: float = 1e-2
    adam_epsilon: float = 1e-08

    # EMA
    use_ema: bool = False
    ema_max_decay: float = 0.9999
    ema_inv_gamma: float = 1.0
    ema_power: float = 3 / 4

    # Hub
    push_to_hub: bool = False
    hub_model_id: str = ''
