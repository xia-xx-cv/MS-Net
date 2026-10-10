from models.msnet.convnext_encoder import convnext_build
from models.msnet.model_utils import *
from models.msnet.blocks import *
import torch
import torch.nn as nn
from torch.nn import ConvTranspose2d
# from .model_utils import DWConvTranspose2d as ConvTranspose2d


class MutualSemanticEnhancingNet(nn.Module):
    _ch_configs = {
        'tiny': (96, 192, 384, 768),
        'small': (96, 192, 384, 768),
        'base': (128, 256, 512, 1024),
        'large': (192, 384, 768, 1536),
    }
    _group_configs = {
        'tiny': (96, 192, 384, 768),  # dw conv for the whole decoder
        'small': (1, 1, 1, 1),  # dw conv for the main decoder, but not for tranaposed conv
        'base': (1, 1, 1, 1),
        'large': (1, 1, 1, 1),
    }

    def __init__(
        self, pretrained_encoder_backbone: bool = True,
            n_class: int = 1, size: str = 'small',
            **kwargs
    ) -> None:
        super(MutualSemanticEnhancingNet, self).__init__()
        self.config = self._ch_configs[size]
        self.c0, self.c1, self.c2, self.c_bottleneck = self.config
        self.groups = self._group_configs[size]

        self.encoder = convnext_build(
            model_size='convnext_'+size,
            pretrained=pretrained_encoder_backbone,** kwargs
        )
        self.downsamplers = list(self.encoder.downsample_layers.children())
        self.encoder_layers = list(self.encoder.stages.children())
        # passing by reference
        self.enc0 = nn.Sequential(self.downsamplers[0], self.encoder_layers[0])
        self.enc0_1x1 = conv_relu(self.c0, self.c0, 1, 0,
                                  groups=self.groups[0])

        self.enc1 = nn.Sequential(self.downsamplers[1], self.encoder_layers[1])
        self.enc1_1x1 = conv_relu(self.c1, self.c1, 1, 0,
                                  groups=self.groups[1])

        self.enc2 = nn.Sequential(self.downsamplers[2], self.encoder_layers[2])
        self.enc2_1x1 = conv_relu(self.c2, self.c2, 1, 0,
                                  groups=self.groups[2])

        self.bottleneck = nn.Sequential(self.downsamplers[3])
        self.bott_1x1 = conv_relu(self.c_bottleneck, self.c_bottleneck, 1, 0,
                                  groups=self.groups[3])

        self.ups_msr = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.ups_msr.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                MutualSemanticRefinement(in_ch=feature, ch=feature, out_ch=feature),
                ConvNextDecoderBlock(feature*2, feature, 1, 1)
            ])
        self.conv_last = nn.Conv2d(self.c0, n_class, kernel_size=1)

        self.boundary_ups = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.boundary_ups.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                ConvNextDecoderBlock(feature*2, feature, 1, 1)
            ])
        self.conv_last_boundary = nn.Conv2d(self.c0, n_class, kernel_size=1)

    @classmethod
    def tiny(cls, **kwargs):
        return cls(config=cls._ch_configs['tiny'], **kwargs)

    @classmethod
    def small(cls, **kwargs):
        return cls(config=cls._ch_configs['small'], **kwargs)

    @classmethod
    def base(cls, **kwargs):
        return cls(config=cls._ch_configs['base'], **kwargs)

    @classmethod
    def large(cls, **kwargs):
        return cls(config=cls._ch_configs['large'], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.size()
        # in paper: 1,2,3
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)

        bottleneck = self.bottleneck(enc2)
        bottleneck = self.bott_1x1(bottleneck)

        x_boundary = self.boundary_ups[0](bottleneck)  # transposed conv
        x_content = self.ups_msr[0](bottleneck) # transposed conv
        enc2 = self.enc2_1x1(enc2)
        """ mutual semantic refinement(msr): 
        [1]boundary-refined content, [2]content-guided boundary
        """
        # enc2: x_s_tilde in Fig.2 and the output of yellow boxes in Fig.1
        enc2 = self.ups_msr[1](x_b=x_boundary, x_c=x_content, x_e=enc2)
        # x_content: light and dark orange boxes in Fig.1
        x_content = torch.cat([x_content, enc2], dim=1)
        # the dark blue boxes (decoders) in Fig.1
        x_content = self.ups_msr[2](x_content) # decoder_c
        # the light and dark violet boxes in Fig. 1
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[1](x_boundary)  # decoder_b

        x_boundary = self.boundary_ups[2](x_boundary) # boundary feat.
        x_content = self.ups_msr[3](x_content) # transposed conv
        enc1 = self.enc1_1x1(enc1)
        enc1 = self.ups_msr[4](x_b=x_boundary, x_c=x_content, x_e=enc1)
        x_content = torch.cat([x_content, enc1], dim=1)
        x_content = self.ups_msr[5](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[3](x_boundary)

        x_boundary = self.boundary_ups[4](x_boundary)
        x_content = self.ups_msr[6](x_content)
        enc0 = self.enc0_1x1(enc0)
        enc0 = self.ups_msr[7](x_b=x_boundary, x_c=x_content, x_e=enc0)
        x_content = torch.cat([x_content, enc0], dim=1)
        x_content = self.ups_msr[8](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[5](x_boundary)

        boundary_out = self.conv_last_boundary(x_boundary)
        content_out = self.conv_last(x_content)

        content_out = nn.functional.interpolate(content_out, size=(H, W), mode="bilinear", align_corners=False)
        boundary_out = nn.functional.interpolate(boundary_out, size=(H, W), mode="bilinear", align_corners=False)
        return content_out, boundary_out


    _ch_configs = {
        'tiny': (96, 192, 384, 768),
        'small': (96, 192, 384, 768),
        'base': (128, 256, 512, 1024),
        'large': (192, 384, 768, 1536),
    }

    def __init__(
            self, pretrained_encoder_backbone: bool = True,
            n_class: int = 1, size: str = 'tiny',
            **kwargs
    ) -> None:
        super(MutualSemanticEnhancingNet, self).__init__()
        self.config = self._ch_configs[size]
        self.c0, self.c1, self.c2, self.c_bottleneck = self.config
        # self.groups = self._group_configs[size]

        self.encoder = convnext_build(
            model_size='convnext_' + size,
            pretrained=pretrained_encoder_backbone, **kwargs
        )
        self.downsamplers = list(self.encoder.downsample_layers.children())
        self.encoder_layers = list(self.encoder.stages.children())

        # OURS: groups and OURS*: groups or not
        self.enc0 = nn.Sequential(self.downsamplers[0], self.encoder_layers[0])
        self.enc0_1x1 = conv_relu(self.c0, self.c0, 1, 0)  # do not use groups=self.groups[0]

        self.enc1 = nn.Sequential(self.downsamplers[1], self.encoder_layers[1])
        self.enc1_1x1 = conv_relu(self.c1, self.c1, 1, 0)  # do not use groups=self.groups[1]

        self.enc2 = nn.Sequential(self.downsamplers[2], self.encoder_layers[2])
        self.enc2_1x1 = conv_relu(self.c2, self.c2, 1, 0)  # do not use groups=self.groups[2]

        self.bottleneck = nn.Sequential(self.downsamplers[3])
        self.bott_1x1 = conv_relu(self.c_bottleneck, self.c_bottleneck, 1, 0)  # do not use groups=self.groups[3]

        self.ups_msr = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.ups_msr.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                MutualSemanticRefinement(in_ch=feature, ch=feature, out_ch=feature),
                ConvNextDecoderBlock(feature*2, feature, 1, 1)
            ])
        self.conv_last = nn.Conv2d(self.c0, n_class, kernel_size=1)

        self.boundary_ups = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.boundary_ups.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                ConvNextDecoderBlock(feature*2, feature, 1, 1)
            ])
        self.conv_last_boundary = nn.Conv2d(self.c0, n_class, kernel_size=1)

    @classmethod
    def tiny(cls, **kwargs):
        return cls(config=cls._ch_configs['tiny'], **kwargs)

    @classmethod
    def small(cls, **kwargs):
        return cls(config=cls._ch_configs['small'], **kwargs)

    @classmethod
    def base(cls, **kwargs):
        return cls(config=cls._ch_configs['base'], **kwargs)

    @classmethod
    def large(cls, **kwargs):
        return cls(config=cls._ch_configs['large'], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.size()
        # in paper: 1,2,3
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)

        bottleneck = self.bottleneck(enc2)
        bottleneck = self.bott_1x1(bottleneck)

        x_boundary = self.boundary_ups[0](bottleneck)  # transposed conv
        x_content = self.ups_msr[0](bottleneck) # transposed conv
        enc2 = self.enc2_1x1(enc2)
        """ mutual semantic refinement(msr):
        [1]boundary-refined content, [2]content-guided boundary
        """
        # enc2: x_s_tilde in Fig.2 and the output of yellow boxes in Fig.1
        enc2 = self.ups_msr[1](x_b=x_boundary, x_c=x_content, x_e=enc2)
        # x_content: light and dark orange boxes in Fig.1
        x_content = torch.cat([x_content, enc2], dim=1)
        # the dark blue boxes (decoders) in Fig.1
        x_content = self.ups_msr[2](x_content) # decoder_c
        # the light and dark violet boxes in Fig. 1
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[1](x_boundary)  # decoder_b

        x_boundary = self.boundary_ups[2](x_boundary) # boundary feat.
        x_content = self.ups_msr[3](x_content) # transposed conv
        enc1 = self.enc1_1x1(enc1)
        enc1 = self.ups_msr[4](x_b=x_boundary, x_c=x_content, x_e=enc1)
        x_content = torch.cat([x_content, enc1], dim=1)
        x_content = self.ups_msr[5](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[3](x_boundary)

        x_boundary = self.boundary_ups[4](x_boundary)
        x_content = self.ups_msr[6](x_content)
        enc0 = self.enc0_1x1(enc0)
        enc0 = self.ups_msr[7](x_b=x_boundary, x_c=x_content, x_e=enc0)
        x_content = torch.cat([x_content, enc0], dim=1)
        x_content = self.ups_msr[8](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[5](x_boundary)

        boundary_out = self.conv_last_boundary(x_boundary)
        content_out = self.conv_last(x_content)

        content_out = nn.functional.interpolate(content_out, size=(H, W), mode="bilinear", align_corners=False)
        boundary_out = nn.functional.interpolate(boundary_out, size=(H, W), mode="bilinear", align_corners=False)
        return content_out, boundary_out


class MutualSemanticEnhancingNet_lite(nn.Module):
    _ch_configs = {
        'tiny': (96, 192, 384, 768),
        'small': (96, 192, 384, 768),
        'base': (128, 256, 512, 1024),
        'large': (192, 384, 768, 1536),
    }
    _group_configs = {
        'tiny': (96, 192, 384, 768),  # dw conv for the whole decoder
        'small': (1, 1, 1, 1),  # dw conv for the main decoder, but not for tranaposed conv
        'base': (1, 1, 1, 1),
        'large': (1, 1, 1, 1),
    }

    def __init__(
        self, pretrained_encoder_backbone: bool = True,
            n_class: int = 1, size: str = 'small',
            **kwargs
    ) -> None:
        super(MutualSemanticEnhancingNet_lite, self).__init__()
        self.config = self._ch_configs[size]
        self.c0, self.c1, self.c2, self.c_bottleneck = self.config
        self.groups = self._group_configs[size]

        self.encoder = convnext_build(
            model_size='convnext_'+size,
            pretrained=pretrained_encoder_backbone,** kwargs
        )
        self.downsamplers = list(self.encoder.downsample_layers.children())
        self.encoder_layers = list(self.encoder.stages.children())
        # passing by reference
        self.enc0 = nn.Sequential(self.downsamplers[0], self.encoder_layers[0])
        self.enc0_1x1 = conv_relu(self.c0, self.c0, 1, 0,
                                  groups=self.groups[0])

        self.enc1 = nn.Sequential(self.downsamplers[1], self.encoder_layers[1])
        self.enc1_1x1 = conv_relu(self.c1, self.c1, 1, 0,
                                  groups=self.groups[1])

        self.enc2 = nn.Sequential(self.downsamplers[2], self.encoder_layers[2])
        self.enc2_1x1 = conv_relu(self.c2, self.c2, 1, 0,
                                  groups=self.groups[2])

        self.bottleneck = nn.Sequential(self.downsamplers[3])
        self.bott_1x1 = conv_relu(self.c_bottleneck, self.c_bottleneck, 1, 0,
                                  groups=self.groups[3])

        self.ups_msr = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.ups_msr.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                MutualSemanticRefinement(in_ch=feature, ch=feature, out_ch=feature),
                ConvNextDecoderBlock_lite(feature*2, feature, 1, 1)
            ])
        self.conv_last = nn.Conv2d(self.c0, n_class, kernel_size=1)

        self.boundary_ups = nn.ModuleList()
        for feature in [self.c2, self.c1, self.c0]:
            self.boundary_ups.extend([
                ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2),
                ConvNextDecoderBlock_lite(feature*2, feature, 1, 1)
            ])
        self.conv_last_boundary = nn.Conv2d(self.c0, n_class, kernel_size=1)

    @classmethod
    def tiny(cls, **kwargs):
        return cls(config=cls._ch_configs['tiny'], **kwargs)

    @classmethod
    def small(cls, **kwargs):
        return cls(config=cls._ch_configs['small'], **kwargs)

    @classmethod
    def base(cls, **kwargs):
        return cls(config=cls._ch_configs['base'], **kwargs)

    @classmethod
    def large(cls, **kwargs):
        return cls(config=cls._ch_configs['large'], **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, C, H, W = x.size()
        # in paper: 1,2,3
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)

        bottleneck = self.bottleneck(enc2)
        bottleneck = self.bott_1x1(bottleneck)

        x_boundary = self.boundary_ups[0](bottleneck)  # transposed conv
        x_content = self.ups_msr[0](bottleneck) # transposed conv
        enc2 = self.enc2_1x1(enc2)
        """ mutual semantic refinement(msr): 
        [1]boundary-refined content, [2]content-guided boundary
        """
        # enc2: x_s_tilde in Fig.2 and the output of yellow boxes in Fig.1
        enc2 = self.ups_msr[1](x_b=x_boundary, x_c=x_content, x_e=enc2)
        # x_content: light and dark orange boxes in Fig.1
        x_content = torch.cat([x_content, enc2], dim=1)
        # the dark blue boxes (decoders) in Fig.1
        x_content = self.ups_msr[2](x_content) # decoder_c
        # the light and dark violet boxes in Fig. 1
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[1](x_boundary)  # decoder_b

        x_boundary = self.boundary_ups[2](x_boundary) # boundary feat.
        x_content = self.ups_msr[3](x_content) # transposed conv
        enc1 = self.enc1_1x1(enc1)
        enc1 = self.ups_msr[4](x_b=x_boundary, x_c=x_content, x_e=enc1)
        x_content = torch.cat([x_content, enc1], dim=1)
        x_content = self.ups_msr[5](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[3](x_boundary)

        x_boundary = self.boundary_ups[4](x_boundary)
        x_content = self.ups_msr[6](x_content)
        enc0 = self.enc0_1x1(enc0)
        enc0 = self.ups_msr[7](x_b=x_boundary, x_c=x_content, x_e=enc0)
        x_content = torch.cat([x_content, enc0], dim=1)
        x_content = self.ups_msr[8](x_content)  # transposed conv
        x_boundary = torch.cat([x_boundary, x_content.detach()], dim=1)
        x_boundary = self.boundary_ups[5](x_boundary)

        boundary_out = self.conv_last_boundary(x_boundary)
        content_out = self.conv_last(x_content)

        content_out = nn.functional.interpolate(content_out, size=(H, W), mode="bilinear", align_corners=False)
        boundary_out = nn.functional.interpolate(boundary_out, size=(H, W), mode="bilinear", align_corners=False)
        return content_out, boundary_out


if __name__=="__main__":
  import torch
  from thop import profile
  x = torch.randn(8,3,224,224)
  model = MutualSemanticEnhancingNet(n_class=3, size='small', pretrained_encoder_backbone=False)
  model.eval()
  input_tensor = torch.randn(1, 3, 224, 224)
  flops, params = profile(model, inputs=(input_tensor,))
  print(f"total Params: {params / (1000*1000):.2f} MB")
  print(f"FLOPs: {flops / (1000*1000*1000):.2f} GFLOPs")

  def count_params(model):
    return sum(p.numel() for p in model.parameters())
  print(f"manually counting params：{count_params(model)/(1000*1000):.2f}")
  # cont, boundary = model(x)
  # print(cont.shape)