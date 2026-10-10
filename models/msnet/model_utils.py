from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_relu(in_channels: int , out_channels: int , kernel: int, padding:int, groups:int=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, groups=groups),
        nn.ReLU(inplace=True),
    )

def conv_gelu(in_channels: int , out_channels: int , kernel: int, padding:int, groups:int=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, groups=groups),
        nn.GELU(approximate="none"),
    )


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # shared MLP
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, out_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x0 = x
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        att = self.sigmoid(avg_out + max_out)
        return self.reduce(x0) * att


class LayerNorm(nn.Module):
    """Implementation of layer normalization for the data format: 
    channels_first. Thus the dimensions of the input and the output will be
    (batch_size, channels, height, width)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x.
    """
    
    def __init__(self, normalized_dim, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_dim))
        self.bias = nn.Parameter(torch.zeros(normalized_dim))
        self.eps = eps
        self.normalized_dim = (normalized_dim, )

    def forward(self, x):
        u = x.mean(1, keepdims = True)
        s = (x-u).pow(2).mean(1, keepdim = True)
        x = (x-u)/ torch.sqrt( s + self.eps)
        x = self.weight[:,None, None]*x + self.bias[:,None, None]
        return x


class LayerNormChannelLast(nn.Module):

    """Implementation of layer normalization for the data format: 
    channels_last. Thus the dimensions of the input and the output will be
    (batch_size, height, width, channels)
    
    Keyword arguments:
    
    arguments -- 
    normalized_dim - dimension of the tensors to be normalized
    eps - epsilon value for Layer Normalization equation

    Return: Normalized value of input tensor x."""

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super(LayerNormChannelLast, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x: Tensor) -> Tensor:
        #Just call layer norm from nn.Functional
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class MutualSemanticRefinement(nn.Module):
  def __init__(self, in_ch: int , ch: int , out_ch: int) -> None:
        super(MutualSemanticRefinement,self).__init__()
        self.f_enc  = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_ch)
            )
        self.wsum = nn.Conv2d(2, 1, 3, padding=1, bias=False)

        self.f_content  = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_ch)
        )
        self.g_content = ChannelAttention(in_ch, out_ch)
        self.f_boundary = nn.Sequential(
            nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_ch)
        )
        self.g_boundary = ChannelAttention(in_ch, out_ch)

        self.convf = nn.Conv2d(in_ch, in_ch, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.gelu = nn.GELU(approximate="none")

  def forward(self, x_b: Tensor , x_c: Tensor, x_e: Tensor) -> Tensor:
    # x_boundary->x_b; x_content->x_c; x_encoder->x_e
    # feature selection
    x_e1 = self.f_enc(x_e)
    avgout = torch.mean(x_e1, dim=1, keepdim=True)
    maxout, _ = torch.max(x_e1, dim=1, keepdim=True)
    x_e2 = self.wsum(torch.cat([avgout, maxout], dim=1))
    g_e = self.sigmoid(x_e2)
    x_e_tilde = x_e1 * g_e

    # boudnary-aware semantic selection
    x_b_tilde = self.g_boundary(self.f_boundary(x_b))
    x_c_tilde = self.g_content(self.f_content(x_c))

    # x_s = self.gelu(x_c_tilde+x_b_tilde)  # x_semantic-->x_s
    x_s = x_c_tilde + x_b_tilde
    g_s = self.sigmoid(self.convf(x_s))
    x_s_tilde = x_s * g_s

    return x_e_tilde + x_s_tilde


class DWConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=2, stride=2):
        super().__init__()
        self.depthwise = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x