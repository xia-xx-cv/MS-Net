import torch.nn as nn
from torch import Tensor
from models.msnet.model_utils import LayerNormChannelLast
from timm.models.layers import DropPath


class ConvNextEncoderBlock(nn.Module):
    """
    Standard ConvNeXt encoder block taken from
    "Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022."
    This implementation is based on Layer normalization. Refer to Fig. 2 of our paper for the definitive structure of this block

    Keyword arguments:
    argument --
    dim - dimensions of the input tensor
    drop_path - drop value of the DropPath() function. If its <=0 then an identity function is chosen instead
    layer_scale_init_values - scaling values

    Return:
     - None
    """

    def __init__(self, dim: int, drop_path: float = 0., layer_scale_init_value: float = 1e-6) -> None:
        super(ConvNextEncoderBlock, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # Depth-wise Convolution
        self.norm = LayerNormChannelLast(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # Pointwise or 1x1 Conv layer
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)),
            requires_grad=True) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """
        forward function

        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return: x : Output torch.Tensor of the encoder block
        """

        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # Permuting Dimensions --- (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)  # Layer Norm with channel dimensions are aligned at the last
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # Permuting Dimensions --- (B, H, W, C) -> (B, C, H, W)

        x = input_x + self.drop_path(x)  # Add the residual connection
        return x


class ConvNextDecoderBlock(nn.Module):
    """
    A straightforward implementation of modified ConvNext blocks used for constructing the decoder network.
    For visualization, refer to Fig. 2.

    Keyword arguments:
    arguments --
    input_dim : Input dimensions of the input torch.Tensor for the first conv 7x7 layer
    output_dim : Output dimensions of all the conv 1x1 layers. It also serves as the input dims of all the
                conv 1x1 layers


    Return: None
    """

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock, self).__init__()

        self.mod_convNext_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=padding, groups=output_dim),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=stride, padding=padding),
            nn.GELU(approximate='none'),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=stride, padding=padding)
        )

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

        self.gelu = nn.GELU(approximate='none')

    def forward(self, x: Tensor) -> Tensor:
        """forward function

        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return:  Output torch.Tensor of the encoder block
        """
        return self.gelu(self.mod_convNext_block(x) + self.conv_skip_connection(x))


class ConvNextDecoderBlock_lite(nn.Module):
    """
    A straightforward implementation of modified ConvNext blocks used for constructing the decoder network.
    For visualization, refer to Fig. 2.

    Keyword arguments:
    arguments --
    input_dim : Input dimensions of the input torch.Tensor for the first conv 7x7 layer
    output_dim : Output dimensions of all the conv 1x1 layers. It also serves as the input dims of all the
                conv 1x1 layers


    Return: None
    """

    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock_lite, self).__init__()

        self.mod_convNext_block = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=padding, groups=output_dim),
            nn.BatchNorm2d(output_dim),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=stride, padding=padding),
            nn.GELU(approximate='none'),
            nn.Conv2d(output_dim, output_dim, kernel_size=1, stride=stride, padding=padding)
        )

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding, groups=output_dim),
            nn.BatchNorm2d(output_dim)
            #  nn.Conv2d(output_dim, output_dim, kernel_size = 1, stride = stride, padding = padding)
        )

        self.gelu = nn.GELU(approximate='none')

    def forward(self, x: Tensor) -> Tensor:
        """forward function

        Keyword arguments:
        argument -- x : Input torch.Tensor
        Return:  Output torch.Tensor of the encoder block
        """
        return self.gelu(self.mod_convNext_block(x) + self.conv_skip_connection(x))


class ConvNextDecoderBlock1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock1, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=3, groups=output_dim),
            nn.BatchNorm2d(output_dim),
        )

        self.pointwise_kan1 = KANLinear(output_dim, output_dim)
        self.act = nn.GELU(approximate='none')
        self.pointwise_kan2 = KANLinear(output_dim, output_dim)

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.conv_skip_connection(x)

        out = self.depthwise_conv(x)  # (B, C, H, W)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)

        out = self.pointwise_kan1(out)
        out = self.act(out)
        out = self.pointwise_kan2(out)

        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return self.act(out + identity)


class ConvNextDecoderBlock1(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock1, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=3, groups=output_dim),
            nn.BatchNorm2d(output_dim),
        )

        self.pointwise_kan1 = KANLinear(output_dim, output_dim)
        self.act = nn.GELU(approximate='none')
        self.pointwise_kan2 = KANLinear(output_dim, output_dim)

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.conv_skip_connection(x)

        out = self.depthwise_conv(x)  # (B, C, H, W)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)

        out = self.pointwise_kan1(out)
        out = self.act(out)
        out = self.pointwise_kan2(out)

        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return self.act(out + identity)


class ConvNextDecoderBlock2(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock2, self).__init__()

        self.depthwise_conv = nn.Sequential(
            # nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=3, groups=output_dim),
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=6, dilation=2, groups=output_dim),
            nn.BatchNorm2d(output_dim),
        )

        # 用 KAN 替换，注意输入是 output_dim，输出也是 output_dim
        self.pointwise_kan = KAN(layers_hidden=[output_dim, output_dim])

        self.act = nn.GELU(approximate='none')

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.conv_skip_connection(x)

        out = self.depthwise_conv(x)  # (B, C, H, W)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)

        out = self.pointwise_kan(out)
        out = self.act(out)

        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return self.act(out + identity)


class ConvNextDecoderBlock1_v2_noatt(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int = 1, padding: int = 1) -> None:
        super(ConvNextDecoderBlock1_v2_noatt, self).__init__()

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=7, stride=stride, padding=6, dilation=2, groups=output_dim),
            nn.BatchNorm2d(output_dim),
        )

        self.pointwise_kan1 = KANLinear(output_dim, output_dim)
        self.act = nn.SiLU()  # Swish 激活
        self.pointwise_kan2 = KANLinear(output_dim, output_dim)

        self.conv_skip_connection = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=padding),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.conv_skip_connection(x)

        out = self.depthwise_conv(x)  # (B, C, H, W)
        B, C, H, W = out.shape
        out = out.permute(0, 2, 3, 1).reshape(B * H * W, C)  # (B*H*W, C)

        out = self.pointwise_kan1(out)
        out = self.act(out)
        out = self.pointwise_kan2(out)

        out = out.view(B, H, W, C).permute(0, 3, 1, 2)  # (B, C, H, W)

        return self.act(out + identity)







