from abc import ABCMeta, abstractmethod


class Trainer(metaclass=ABCMeta):
    def __init__(self, weight_dtype, accelerator, logger, cfg):
        self.weight_dtype = weight_dtype
        self.accelerator = accelerator
        self.logger = logger
        self.cfg = cfg

    @abstractmethod
    def init_modules(self,
                     enable_xformer: bool = False,
                     gradient_checkpointing: bool = False):
        pass

    @abstractmethod
    def init_optimizers(self, train_batch_size):
        pass

    @abstractmethod
    def init_lr_schedulers(self, gradient_accumulation_steps, num_epochs):
        pass

    def set_dataset(self, dataset, train_dataloader):
        self.dataset = dataset
        self.train_dataloader = train_dataloader

    @abstractmethod
    def prepare_modules(self):
        pass

    @abstractmethod
    def models_to_train(self):
        pass

    @abstractmethod
    def training_step(self, global_step, batch) -> dict:
        pass

    @abstractmethod
    def validate(self, epoch, global_step):
        pass

    @abstractmethod
    def save_pipeline(self):
        pass

    @abstractmethod
    def save_model_hook(self, models, weights, output_dir):
        pass

    @abstractmethod
    def load_model_hook(self, models, input_dir):
        pass


def create_trainer(type, weight_dtype, accelerator, logger, cfg_dict) -> Trainer:
    from ddpm_trainer import DDPMTrainer
    from sd_lora_trainer import LoraTrainer

    __TYPE_CLS_DICT = {
        'ddpm': DDPMTrainer,
        'lora': LoraTrainer
    }

    return __TYPE_CLS_DICT[type](weight_dtype, accelerator, logger, cfg_dict)
