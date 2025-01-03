from enum import Enum

from datasets import load_dataset
from torchvision import transforms


class DataKey(Enum):
    IMAGE = 0
    INT_LABLE = 1
    TEXT = 2


def create_dataset(
    resolution,
    dataset_name=None,
    dataset_config_name=None,
    data_dir=None,
    data_files=None,
    dataset_key_map={DataKey.IMAGE: 'image'},
    split='train',
    center_crop=False,
    random_flip=False,
    cache_dir=None
):
    if dataset_name is not None:
        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            split=split,
        )
    elif data_files is not None:
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            split=split
        )
    elif data_dir is not None:
        dataset = load_dataset(
            "imagefolder",
            data_dir=data_dir,
            cache_dir=cache_dir,
            split=split)
    else:
        raise ValueError(
            '`dataset_name`, `data_files`, or `data_dir` must not be None')

    augmentations = transforms.Compose(
        [
            transforms.Resize(
                resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(
                resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform_images(examples):
        result_dict = {}

        if DataKey.IMAGE in dataset_key_map:
            dataset_image_key = dataset_key_map[DataKey.IMAGE]
            images = [augmentations(image.convert("RGB"))
                      for image in examples[dataset_image_key]]
            result_dict[DataKey.IMAGE] = images
        return result_dict

    dataset.set_transform(transform_images)

    return dataset
