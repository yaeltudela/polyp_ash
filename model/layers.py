import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from timm.models.layers import trunc_normal_


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm, add_pix=False):
        super().__init__()
        self.input_resolution = input_resolution, input_resolution
        self.dim = dim
        self.expansion_f = 2
        self.num_features = dim // self.expansion_f
        self.expansion = nn.Linear(dim, self.expansion_f * dim, bias=False)
        self.norm = norm_layer(self.num_features)
        self.add_pix = add_pix

        if self.add_pix:
            self.join = nn.Linear(2 * self.num_features, self.num_features, bias=False)
            self.pix = nn.PixelShuffle(self.expansion_f)
            self.norm = norm_layer(2 * self.num_features)

        self.apply(self.init_weights)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape

        assert L == H * W, f"input feature has wrong size - L={L}, H={H}, W={W}"

        x = self.expansion(x)
        x = rearrange(x, 'b (h w) c -> b h w c', h=H, w=W)
        c = x.size(-1)

        x_orig = rearrange(x, 'b h w (c p1 p2) -> b (h p1 w p2) c', c=c // (self.expansion_f ** 2), p1=self.expansion_f,
                           p2=self.expansion_f)

        if self.add_pix:
            x_p = rearrange(x, 'b h w c -> b c h w')
            x_p = self.pix(x_p)
            x_p = rearrange(x_p, 'b c h w-> b (h w) c')

            x = torch.cat([x_orig, x_p], dim=-1)
            x = self.norm(x)
            x = self.join(x)
        else:
            x = x_orig
            x = self.norm(x)

        return x

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}, expansion_factor={self.expansion_f} \n " \
               f"input -> output= B {self.input_resolution[0] * self.input_resolution[1]} {self.dim} ==> B " \
               f"{self.expansion_f ** 2 * self.input_resolution[0] * self.input_resolution[1]} " \
               f"{(self.expansion_f * self.dim) // self.expansion_f ** 2}"


def init_normal(m, mean=0., std=1.):
    if hasattr(m, 'weight') and m.weight is not None:
        nn.init.normal_(m.weight, mean=mean, std=std)
    if hasattr(m, 'bias') and m.bias is not None:
        nn.init.constant_(m.bias, 0)


class PPM(nn.Module):

    def __init__(self, pool_scales, in_channels, out_channels):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling_scales = nn.ModuleList([])
        for pool_scale in pool_scales:
            self.pooling_scales.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    Conv2d(
                        self.in_channels,
                        self.out_channels,
                        1,
                        norm=nn.BatchNorm2d,
                        activation=F.relu,
                    )))

        self.bottleneck = Conv2d(self.out_channels * len(self.pool_scales) + self.in_channels, self.out_channels,
                                 kernel_size=3, padding=1, norm=nn.BatchNorm2d, activation=F.relu)

        self.apply(self._init_weights)

    def forward(self, x):
        """Forward function."""
        ppm_outs = [x]
        for ppm in self.pooling_scales:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=False)
            ppm_outs.append(upsampled_ppm_out)

        ppm_out = self.bottleneck(torch.cat(ppm_outs, dim=1))
        return ppm_out

    def _init_weights(self, m):
        # uniform for Convs
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class Upsample(nn.Module):
    def __init__(self, inplanes, planes, factor=2, mode='bilinear', kernel_size=5, use_bn=True):
        super(Upsample, self).__init__()
        align_corners = True if mode != 'nearest' else None
        padding = (kernel_size - 1) // 2  # assumes strides == 1
        self.up = nn.Upsample(scale_factor=factor, mode=mode, align_corners=align_corners)
        # self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, padding=0)

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        if use_bn:
            self.bn = nn.BatchNorm2d(planes)
        self.factor = factor
        self.use_bn = use_bn

        self.apply(self._init_weights)

    def forward(self, x):
        # x = F.interpolate(x, scale_factor=self.factor, mode='bilinear', align_corners=False)
        x = self.up(x)
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn(x)
        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class Conv2d(torch.nn.Conv2d):
    """
    A wrapper around :class:`torch.nn.Conv2d` to support empty inputs and more features.
    """

    def __init__(self, *args, **kwargs):
        """
        Extra keyword arguments supported in addition to those in `torch.nn.Conv2d`:
        Arguments:
            norm (nn.Module, optional): a normalization layer
            activation (callable(Tensor) -> Tensor): a callable activation function
        It assumes that norm layer is used before activation.
        """
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)

        if norm:
            kwargs.update({'bias': False})

        super().__init__(*args, **kwargs)

        self.is_ln = False
        if norm is not None:
            self.norm = norm(self.out_channels)
            if isinstance(norm, nn.LayerNorm):
                self.is_ln = True
        else:
            self.norm = None
        self.activation = activation

    def forward(self, x):

        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        if self.norm is not None:
            if self.is_ln:
                x = x.permute(0, 2, 3, 1)  # b h w c
                x = self.norm(x)
                x = x.permute(0, 3, 1, 2)  # b c h w
            else:
                x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def extra_repr(self):
        return super(Conv2d, self).extra_repr() + f" norm: {self.norm} - activation: {self.activation.__name__ if self.activation is not None else 'None'}"
