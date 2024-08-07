import matplotlib.pyplot as plt
import torch

__all__ = ["show_outputs_w_uncertainties"]


def show_outputs_w_uncertainties(x_a: torch.Tensor,
                                 x_b: torch.Tensor,
                                 y_b: torch.Tensor,
                                 map_mu: torch.Tensor,
                                 map_alpha: torch.Tensor,
                                 map_beta: torch.Tensor,
                                 map_error: torch.Tensor,
                                 map_var: torch.Tensor,) -> None:
    """

    :param x_a:
    :param x_b:
    :param y_b:
    :param map_mu:
    :param map_alpha:
    :param map_beta:
    :param map_error:
    :param map_var:
    :return:
    """
    plt.figure(figsize=(30, 10))
    #
    plt.subplot(1, 4, 1)
    plt.imshow(x_a.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 4, 2)
    plt.imshow(y_b.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 4, 3)
    plt.imshow(map_alpha.transpose(0, 2).transpose(0, 1), cmap="inferno")
    plt.clim(0, 0.1)
    plt.axis("off")
    #
    plt.subplot(1, 4, 4)
    plt.imshow(map_error, cmap="jet")
    plt.clim(0, 0.01)
    plt.axis("off")
    #
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
    #
    plt.figure(figsize=(30, 10))
    plt.subplot(1, 4, 1)
    plt.imshow(x_b.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 4, 2)
    plt.imshow((0.6 * map_mu + 0.4 * y_b).clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    plt.show()
    #
    plt.subplot(1, 4, 3)
    plt.imshow(map_beta.transpose(0, 2).transpose(0, 1), cmap="cividis")
    plt.clim(0.45, 0.75)
    plt.axis("off")
    # show uncertainty map
    plt.subplot(1, 4, 4)
    plt.imshow(map_var.transpose(0, 2).transpose(0, 1), cmap="hot")
    plt.clim(0, 0.15)
    plt.axis("off")
    #
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    plt.show()
