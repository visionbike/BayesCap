import math
import numpy as np
import torch
import kornia

__all__ = [
    "compute_correlation_coefficient",
    "compute_image_mae",
    "compute_image_mse",
    "compute_image_psnr",
    "compute_image_ssim",
    "compute_uce"
]


def compute_correlation_coefficient(list_err: list, list_var: list) -> np.ndarray:
    """
    Compute the correlation coefficient between two lists.

    :param list_err:
    :param list_var:
    :return:
    """
    mat_covar = np.corrcoef(np.array(list_err), np.array(list_var))
    return mat_covar[0][1] / (math.sqrt(mat_covar[0][0]) * math.sqrt(mat_covar[1][1]))


def compute_image_mae(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean absolute difference between two images.

    :param x1:
    :param x2:
    :return:
    """
    m = torch.abs(x1 - x2).mean()
    return m


def compute_image_mse(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the mean squared difference between two images.

    :param x1:
    :param x2:
    :return:
    """
    m = torch.pow(torch.abs(x1 - x2), 2).mean()
    return m


def compute_image_psnr(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the PSNR between two images.

    :param x1:
    :param x2:
    :return:
    """
    m = kornia.metrics.psnr(x1, x2, 1)
    return m


def compute_image_ssim(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Compute the SSIM between two images.

    :param x1:
    :param x2:
    :return:
    """
    m = kornia.metrics.ssim(x1.unsqueeze(0), x2.unsqueeze(0), 5).mean()
    return m


def compute_uce(list_err: list, list_var: list, num_bins: int = 100) -> [list, float]:
    """
    Compute the UCE between two lists.

    :param list_err:
    :param list_var:
    :param num_bins:
    :return:
    """
    err_min = np.min(list_err)
    err_max = np.max(list_err)
    err_len = (err_max - err_min) / num_bins
    num_points = len(list_err)
    #
    bin_stats = {}
    for i in range(num_bins):
        bin_stats[i] = {
            "start_idx": err_min + i * err_len,
            "end_idx": err_min + (i + 1) * err_len,
            "num_points": 0,
            "mean_err": 0,
            "mean_var": 0,
        }
    #
    for e, v in zip(list_err, list_var):
        for i in range(num_bins):
            if bin_stats[i]["start_idx"] <= e < bin_stats[i]["end_idx"]:
                bin_stats[i]["num_points"] += 1
                bin_stats[i]["mean_err"] += e
                bin_stats[i]["mean_var"] += v
    #
    uce = 0.
    eps = 1e-8
    for i in range(num_bins):
        bin_stats[i]["mean_err"] /= bin_stats[i]["num_points"] + eps
        bin_stats[i]["mean_var"] /= bin_stats[i]["num_points"] + eps
        bin_stats[i]["uce_bin"] = (bin_stats[i]["num_points"] / num_points) * (np.abs(bin_stats[i]["mean_err"] - bin_stats[i]["mean_var"]))
        uce += bin_stats[i]["uce_bin"]
    #
    list_x, list_y = [], []
    for i in range(num_bins):
        if bin_stats[i]["num_points"] > 0:
            list_x.append(bin_stats[i]["mean_err"])
            list_y.append(bin_stats[i]["mean_var"])
    #
    return bin_stats, uce
