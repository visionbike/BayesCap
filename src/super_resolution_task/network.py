import torch
import torch.nn as nn
import torch.nn.functional as nfn

__all__ = [
    "SRGenerator",
    "SRDiscriminator",
    "SRBayesCap"
]


class ResidualConvBlock(nn.Module):
    """
    Implements residual conv function.
    """

    def __init__(self, channels: int) -> None:
        """

        :param channels: Number of channels in the input image.
        """
        super().__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        z = self.rcb(x)
        z = torch.add(z, identity)
        return z


class SRDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96 or (3) x 84 x 84
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        z = torch.flatten(z, 1)
        z = self.classifier(z)
        return z


class SRGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # first conv layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        # features trunk blocks
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        # second conv layer
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # upscale conv block
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        # output layer
        self.conv_block3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor, dop: float | None = None) -> torch.Tensor:
        if not dop:
            return self._forward_impl(x)
        else:
            return self._forward_w_dop_impl(x, dop)

    # support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        z1 = self.conv_block1(x)
        z = self.trunk(z1)
        z2 = self.conv_block2(z)
        z = torch.add(z1, z2)
        z = self.upsampling(z)
        z = self.conv_block3(z)
        return z

    def _forward_w_dop_impl(self, x: torch.Tensor, dop: float | None) -> torch.Tensor:
        z1 = self.conv_block1(x)
        z = self.trunk(z1)
        z2 = nfn.dropout2d(self.conv_block2(z), p=dop)
        z = torch.add(z1, z2)
        z = self.upsampling(z)
        z = self.conv_block3(z)
        return z

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


# BayesCap
class SRBayesCap(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3) -> None:
        """

        :param in_channels:
        :param out_channels:
        """
        super().__init__()
        # first conv layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        # features trunk blocks
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        # second conv layer
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # output layer
        self.conv_block3_mu = nn.Conv2d(64, out_channels, kernel_size=9, stride=1, padding=4)
        self.conv_block3_alpha = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
        )
        self.conv_block3_beta = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
        )
        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z1 = self.conv_block1(x)
        z = self.trunk(z1)
        z2 = self.conv_block2(z)
        z = z1 + z2
        z_mu = self.conv_block3_mu(z)
        z_alpha = self.conv_block3_alpha(z)
        z_beta = self.conv_block3_beta(z)
        return z_mu, z_alpha, z_beta

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class SRBayesCap_noID(nn.Module):
    def __init__(self, in_channels: int = 3) -> None:
        """

        :param in_channels:
        """
        super().__init__()
        # first conv layer
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
        )
        # features trunk blocks
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        # second conv layer
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        # third conv layer
        self.conv_block3_alpha = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
        )
        self.conv_block3_beta = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
        )
        # initialize neural network weights
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self._forward_impl(x)

    # support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        z1 = self.conv_block1(x)
        z = self.trunk(z1)
        z2 = self.conv_block2(z)
        z = z1 + z2
        z_alpha = self.conv_block3_alpha(z)
        z_beta = self.conv_block3_beta(z)
        return z_alpha, z_beta

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
