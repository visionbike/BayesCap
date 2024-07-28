from torch import Tensor
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as vfn
from .utils import *

__all__ = ["SRDataset"]


class SRDataset(Dataset):
    """
    Customize the dataset loading function and prepare low/high resolution image data in advanced
    """

    def __init__(self, data_root: str, image_size: int | tuple[int, int], upscale_factor: int, mode: str) -> None:
        """

        :param data_root: training data set address.
        :param image_size: high resolution image size.
        :param upscale_factor: image magnification.
        :param mode: data set loading method, the training data set is for data enhancement,
                    and the verification data set is not for data enhancement.
        """
        super().__init__()
        self.image_paths = [path for path in Path(data_root).iterdir() if path.is_file()]
        #
        if mode == "train":
            self.hr_transforms = transforms.Compose([
                transforms.RandomCrop(image_size, pad_if_needed=True),
                transforms.RandomRotation(90),
                transforms.RandomHorizontalFlip(0.5),
            ])
        else:
            self.hr_transforms = transforms.Resize(image_size)
        #
        self.lr_transforms = transforms.Resize(
            (image_size[0] // upscale_factor, image_size[1] // upscale_factor),
            interpolation=vfn.InterpolationMode.BICUBIC,
            antialias=True
        )

    def __getitem__(self, batch_index: int) -> tuple[Tensor, ...]:
        # read a batch of image data
        image = Image.open(str(self.image_paths[batch_index]))
        # transform image
        hr_image = self.hr_transforms(image)
        lr_image = self.lr_transforms(hr_image)
        # convert image data into Tensor stream format (PyTorch).
        # note: the range of input and output is between [0, 1]
        lr_tensor = image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = image2tensor(hr_image, range_norm=False, half=False)
        #
        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.image_paths)
