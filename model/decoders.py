from abc import abstractmethod

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F

from model.layers import PatchExpand, init_normal, PPM, Upsample, Conv2d
from model.fpn import FPN
from model.swin import trunc_normal_


class SwinAdapterDecoder(nn.Module):
    def __init__(self, backbone, fpn, num_classes, head_name='base'):
        super().__init__()
        self.backbone = backbone
        self.fpn = fpn
        self.head_name = head_name
        self.num_classes = num_classes

    def forward(self, x):
        out, intermediate = self.backbone(x)
        x = self.fpn(intermediate)

        out, x = self.forward_decoder(out, x)

        return out, x

    @abstractmethod
    def forward_decoder(self, backbone_output, fpn_outputs):
        pass

    def extra_repr(self) -> str:
        return "".join([f'{k}: {v}\n' for k, v in self.__dict__.items() if k != '_modules'])

    @torch.jit.ignore
    def no_weight_decay(self):
        return self.backbone.no_weight_decay()

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        parent = self.backbone.no_weight_decay_keywords()
        params = {'temperature_scale'}
        return parent.union(params)


class SwinFPN(SwinAdapterDecoder):
    def __init__(self, backbone, fpn, num_classes, use_aux=False, head_name='refine'):
        super().__init__(backbone, fpn, num_classes)
        im_size = self.backbone.image_width
        in_resolution = im_size // self.backbone.patch_width
        self.aux_representation = use_aux

        self.upsamples = nn.ModuleList([
            nn.ModuleList([
                nn.Identity() if not self.aux_representation else Conv2d(self.fpn.out_channels, num_classes,
                                                                         kernel_size=3),
                nn.Upsample(scale_factor=in_resolution // (im_size // s), mode='nearest'),

            ])
            for f, s in self.backbone._out_features
        ])

        self.upsamples = self.upsamples[::-1]

        self.fuse = Conv2d(fpn.out_channels * len(fpn._out_features), fpn.out_channels, kernel_size=1, stride=1,
                           padding=0, norm=nn.BatchNorm2d)

        self.head = build_head(head_name, self.fpn.out_channels, num_classes,
                               hidden_channels=self.fpn.out_channels // 2)

        self.init_decoder_weights()

    def forward_decoder(self, backbone_output, fpn_outputs):

        fuse = []
        aux = []
        for feat, (cam, upsample) in zip(fpn_outputs, self.upsamples):
            aux.append(cam(feat))
            a = upsample(feat)
            fuse.append(a)
        outs = torch.cat(fuse, dim=1)

        x = self.fuse(outs)
        x = self.head(x)

        return backbone_output, x

    def init_decoder_weights(self):
        self.upsamples.apply(self._init_weights)
        self.fuse.apply(self._init_weights)

    def _init_weights(self, m):
        # uniform for Convs
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SwinExpand(SwinAdapterDecoder):
    def __init__(self, backbone, fpn, num_classes, head_name='refine'):
        super().__init__(backbone, fpn, num_classes)

        im_size = self.backbone.image_width
        self.hw = int(im_size // (self.backbone._out_features[0][1]))
        out_channels = self.backbone._out_features[0][0]

        self.upsamples = nn.ModuleList([
            PatchExpand((im_size // stride), dim) for dim, stride in self.backbone._out_features[::-1]
        ])
        self.upsamples[-1] = nn.Identity()

        self.to_patch_expand = Rearrange('b c h w -> b (h w) c')
        in_features = self.backbone._out_features[0][0]
        self.fuse = nn.Sequential(
            Rearrange('b (h w) c -> b c h w', h=self.hw, w=self.hw),
            Conv2d(in_features, out_channels, kernel_size=1, stride=1, padding=0, norm=nn.BatchNorm2d)
        )
        self.head = build_head(head_name, out_channels, num_classes, hidden_channels=out_channels)

        self.init_decoder_weights()

    def forward_decoder(self, backbone_output, fpn_outputs):
        feat = self.to_patch_expand(fpn_outputs[0])

        prev_features = self.upsamples[0][-1](feat)
        for feat, expand_patches in zip(fpn_outputs[1:], self.upsamples[1:]):
            feat = self.to_patch_expand(feat)
            prev_features = feat + prev_features
            prev_features = expand_patches(prev_features)

        x = self.fuse(prev_features)

        x = self.head(x)

        return backbone_output, x

    def init_decoder_weights(self):
        self.upsamples.apply(self._init_weights)
        self.fuse.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SwinSegformer(SwinAdapterDecoder):

    def __init__(self, backbone, fpn, num_classes, head_name='refine'):
        super().__init__(backbone, fpn, num_classes)

        im_size = self.backbone.image_width
        in_resolution = im_size // self.backbone.patch_width

        out_channels = fpn.out_channels
        self.upsamples = nn.ModuleList([
            nn.Sequential(
                Conv2d(out_channels, out_channels=out_channels, kernel_size=1, activation=F.relu),
                nn.Upsample(scale_factor=in_resolution // (im_size // s), mode='bilinear')
            )
            for f, s in self.backbone._out_features
        ])[::-1]

        self.fuse = Conv2d(fpn.out_channels * len(fpn._out_features), fpn.out_channels, kernel_size=1, stride=1,
                           padding=0, norm=nn.BatchNorm2d, activation=F.relu)

        self.head = build_head(head_name, fpn.out_channels, num_classes, hidden_channels=None)

        self.init_decoder_weights()

    def forward_decoder(self, backbone_output, fpn_outputs):
        outs = []
        for feat, upsample in zip(fpn_outputs, self.upsamples):
            a = upsample(feat)
            outs.append(a)

        outs = torch.cat(outs, dim=1)

        x = self.fuse(outs)
        x = self.head(x)

        return backbone_output, x

    def init_decoder_weights(self):
        self.upsamples.apply(self._init_weights)
        self.fuse.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class SwinFPNUperNet(SwinFPN):
    def __init__(self, backbone, fpn, num_classes, head_name='refine'):
        super().__init__(backbone, fpn, num_classes, )

        out_channels = 256
        self.fpn = FPN(self.backbone._out_features, out_channels, norm=nn.BatchNorm2d, lateral_activation=F.relu,
                       output_activation=F.relu)

        ppm = PPM(pool_scales=(1, 2, 3, 6), in_channels=self.backbone._out_features[-1][0], out_channels=out_channels)

        self.fpn.lateral_convs[0] = ppm
        self.fpn.output_convs[0] = nn.Identity()

        self.fuse = Conv2d(self.fpn.out_channels * len(self.fpn._out_features), self.fpn.out_channels, kernel_size=3,
                           stride=1, padding=1, norm=nn.BatchNorm2d, activation=F.relu)

        self.head = build_head(head_name, self.fpn.out_channels, num_classes,
                               hidden_channels=self.fpn.out_channels // 2)
        self.init_decoder_weights()

    def init_decoder_weights(self):
        self.fuse.apply(self._init_weights)


def build_head(head_name, in_channels, num_classes, hidden_channels):
    num_up = 1
    if head_name == 'base':
        return BaseHead(in_channels, num_classes, num_upsamples=num_up, hidden_channels=hidden_channels, p_drop=0.1)
    else:
        raise NotImplemented('Head invalid')


class BaseHead(nn.Sequential):
    def __init__(self, in_channels, num_classes, num_upsamples=2, hidden_channels=None, p_drop=0.1):

        hidden_channels = hidden_channels or in_channels

        upsamples_layers = []
        for i in range(num_upsamples):
            if i == 0:
                upsamples_layers.append(Upsample(in_channels, hidden_channels, factor=2))
            else:
                upsamples_layers.append(Upsample(hidden_channels, hidden_channels, factor=2))

        if num_upsamples == 0:
            hidden_channels = in_channels

        modules = [
            *upsamples_layers,
            nn.Dropout2d(p=p_drop),
            Conv2d(hidden_channels, num_classes, kernel_size=1, padding=0)
        ]

        super().__init__(*modules)
        init_normal(self[-1], mean=0, std=0.01)
