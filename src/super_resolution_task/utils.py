import random
import math
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.transforms import functional as vfn
random.seed(0)

__all__ = [
    "image2tensor",
    "random_mask"
]


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    """
    Convert PIL.Image to Tensor.

    :param image: The image data read by PIL.Image.
    :param range_norm: Scale [0, 1] data to between [-1, 1].
    :param half:  Whether to convert torch.float32 similarly to torch.half type.
    :return: Normalized image data
    """
    tensor = vfn.to_tensor(image)
    if range_norm:
        tensor = tensor.mul_(2.0).sub_(1.0)
    if half:
        tensor = tensor.half()
    return tensor


def random_mask(num_batch: int = 1, mask_shape: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """

    :param num_batch:
    :param mask_shape:
    :return:
    """
    list_mask = []
    for _ in range(num_batch):
        # rectangle mask
        image_height = mask_shape[0]
        image_width = mask_shape[1]
        max_delta_height = image_height // 8
        max_delta_width = image_width // 8
        height = image_height // 4
        width = image_width // 4
        max_t = image_height - height
        max_l = image_width - width
        t = random.randint(0, max_t)
        l = random.randint(0, max_l)
        # bbox = (t, l, height, width)
        h = random.randint(0, max_delta_height // 2)
        w = random.randint(0, max_delta_width // 2)
        mask = torch.zeros((1, 1, image_height, image_width))
        mask[:, :, (t + h): (t + height - h), (l + w): (l + width - w)] = 1
        rect_mask = mask
        # brush mask
        min_num_vertex = 4
        max_num_vertex = 12
        mean_angle = 2 * math.pi / 5
        angle_range = 2 * math.pi / 15
        min_width = 12
        max_width = 40
        H, W = image_height, image_width
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new("L", (W, H), 0)
        #
        for _ in range(np.random.randint(1, 4)):
            num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
            angle_min = mean_angle - np.random.uniform(0, angle_range)
            angle_max = mean_angle + np.random.uniform(0, angle_range)
            angles = []
            vertex = []
            for i in range(num_vertex):
                if i % 2 == 0:
                    angles.append(2 * math.pi - np.random.uniform(angle_min, angle_max))
                else:
                    angles.append(np.random.uniform(angle_min, angle_max))
            h, w = mask.size
            vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
            for i in range(num_vertex):
                r = np.clip(np.random.normal(loc=average_radius, scale=average_radius // 2), 0, 2 * average_radius)
                new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
                new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
                vertex.append((int(new_x), int(new_y)))
            #
            draw = ImageDraw.Draw(mask)
            width = int(np.random.uniform(min_width, max_width))
            draw.line(vertex, fill=255, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=255)
        #
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.normal() > 0:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        #
        mask = transforms.ToTensor()(mask)
        mask = mask.reshape((1, 1, H, W))
        brush_mask = mask
        mask = torch.cat([rect_mask, brush_mask], dim=1).max(dim=1, keepdim=True)[0]
        list_mask.append(mask)
    mask = torch.cat(list_mask, dim=0)
    return mask
