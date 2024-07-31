import matplotlib.pyplot as plt
import torch

__all__ = ["show_outputs_w_uncertainties"]


def show_outputs_w_uncertainties(x_a: torch.Tensor,
                                 x_b: torch.Tensor,
                                 y_b: torch.Tensor,
                                 y_b_var: torch.Tensor,
                                 elim: tuple[float, float] = (0, 0.01),
                                 ulim: tuple[float, float] = (0, 0.15)) -> None:
    """

    :param x_a:
    :param x_b:
    :param y_b:
    :param y_b_var:
    :param elim:
    :param ulim:
    :return:
    """
    plt.figure(figsize=(30, 10))
    #
    plt.subplot(1, 5, 1)
    plt.imshow(x_a.to("cpu").data.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 5, 2)
    plt.imshow(x_b.to("cpu").data.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 5, 3)
    plt.imshow(y_b.to("cpu").data.clip(0, 1).transpose(0, 2).transpose(0, 1))
    plt.axis("off")
    #
    plt.subplot(1, 5, 4)
    error_map = torch.mean(torch.pow(torch.abs(y_b - x_b), 2), dim=0).to("cpu").data.unsqueeze(0)
    print("error", error_map.min(), error_map.max())
    plt.imshow(error_map.transpose(0, 2).transpose(0, 1), cmap="jet")
    plt.clim(elim[0], elim[1])
    plt.axis("off")
    #
    plt.subplot(1, 5, 5)
    print("uncer", y_b_var.min(), y_b_var.max())
    plt.imshow(y_b_var.to("cpu").data.transpose(0, 2).transpose(0, 1), cmap="hot")
    plt.clim(ulim[0], ulim[1])
    plt.axis("off")
    #
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
