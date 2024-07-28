import torch
import torch.nn as nn
import torch.nn.functional as nfn
import torchvision.models as models

__all__ = [
    "ContentLoss",
    "GenGaussLoss",
    "TempCombLoss"
]


class ContentLoss(nn.Module):
    """
    Constructs a content loss function based on the VGG19 network.
    Using high-level feature mapping layers from the latter layers will focus more on the texture content of the image.
    Paper reference list:
    - Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network paper. Link: https://arxiv.org/pdf/1609.04802.pdf.
    - ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks paper. Link: https://arxiv.org/pdf/1809.00219.pdf.
    - Perceptual Extreme Super Resolution Network with Receptive Field Block paper. Link: https://arxiv.org/pdf/2005.12597.pdf.
    """

    def __init__(self) -> None:
        super().__init__()
        # load the VGG19 model trained on the ImageNet dataset
        vgg19 = models.vgg19(pretrained=True).eval()
        # extract the thirty-sixth layer output in the VGG19 model as the content loss
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])
        # freeze model parameters
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False
        # the preprocessing method of the input data. This is the VGG model preprocessing method of the ImageNet dataset
        self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        # standardized operations
        sr = sr.sub(self.mean).div(self.std)
        hr = hr.sub(self.mean).div(self.std)
        # find the feature map difference between the two images
        loss = nfn.l1_loss(self.feature_extractor(sr), self.feature_extractor(hr))
        #
        return loss


class GenGaussLoss(nn.Module):
    def __init__(self,
                 reduction: str = "mean",
                 alpha_eps: float = 1e-4, beta_eps: float = 1e-4,
                 resi_min: float = 1e-4, resi_max: float = 1e3) -> None:
        """

        :param reduction:
        :param alpha_eps:
        :param beta_eps:
        :param resi_min:
        :param resi_max:
        """
        super().__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max

    def forward(self, mean: torch.Tensor, one_over_alpha: torch.Tensor, beta: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        one_over_alpha1 = one_over_alpha + self.alpha_eps
        beta1 = beta + self.beta_eps
        #
        resi = torch.abs(mean - target)
        # resi = torch.pow(resi*one_over_alpha1, beta1).clamp(min=self.resi_min, max=self.resi_max)
        resi = (resi * one_over_alpha1 * beta1).clamp(min=self.resi_min, max=self.resi_max)
        # check if resi has nans
        if torch.sum(resi != resi) > 0:
            raise ValueError("resi has nans!!")
        #
        log_one_over_alpha = torch.log(one_over_alpha1)
        log_beta = torch.log(beta1)
        lgamma_beta = torch.lgamma(torch.pow(beta1, -1))
        #
        if torch.sum(log_one_over_alpha != log_one_over_alpha) > 0:
            print("log_one_over_alpha has nan")
        if torch.sum(lgamma_beta != lgamma_beta) > 0:
            print("lgamma_beta has nan")
        if torch.sum(log_beta != log_beta) > 0:
            print("log_beta has nan")
        #
        loss = resi - log_one_over_alpha + lgamma_beta - log_beta
        #
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            raise ValueError("Reduction not supported")


class TempCombLoss(nn.Module):
    def __init__(self,
                 reduction: str = "mean",
                 alpha_eps: float = 1e-4, beta_eps: float = 1e-4,
                 resi_min: float = 1e-4, resi_max: float = 1e3) -> None:
        """

        :param reduction:
        :param alpha_eps:
        :param beta_eps:
        :param resi_min:
        :param resi_max:
        """
        super().__init__()
        self.reduction = reduction
        self.alpha_eps = alpha_eps
        self.beta_eps = beta_eps
        self.resi_min = resi_min
        self.resi_max = resi_max
        #
        self.loss_GenGauss = GenGaussLoss(reduction=self.reduction,
                                          alpha_eps=self.alpha_eps, beta_eps=self.beta_eps,
                                          resi_min=self.resi_min, resi_max=self.resi_max)
        self.loss_l1 = nn.L1Loss(reduction=self.reduction)

    def forward(self,
                mean: torch.Tensor,
                one_over_alpha: torch.Tensor,
                beta: torch.Tensor,
                target1: torch.Tensor,
                target2: torch.Tensor,
                t1: float, t2: float) -> torch.Tensor:
        # target1 is the base model output for identity mapping
        # target2 is the ground truth for the GenGauss loss
        l1 = self.loss_l1(mean, target1)
        l2 = self.loss_GenGauss(mean, one_over_alpha, beta, target2)
        loss = t1 * l1 + t2 * l2
        #
        return loss
