import numpy as np
from torch import Tensor
from torchvision.transforms import functional as vfn

__all__ = ["image2tensor"]


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> Tensor:
    """
    Convert PIL.Image to Tensor.

    :param image: The image data read by PIL.Image.
    :param range_norm: Scale [0, 1] data to between [-1, 1].
    :param half:  Whether to convert torch.float32 similarly to torch.half type.
    :return: Normalized image data
    """
    tensor = vfn.to_tensor(image)
    #
    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()
    #
    return tensor
