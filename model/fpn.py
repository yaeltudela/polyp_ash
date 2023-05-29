# Copyright (c) Facebook, Inc. and its affiliates.
import math
import torch
import torch.nn.functional as F
from torch import nn

from model.layers import Conv2d


class FPN(nn.Module):
    """
    This module implements :paper:`FPN`.
    It creates pyramid features built on top of some input feature maps.
    """

    _fuse_type: torch.jit.Final[str]

    def __init__(self, input_channels_stride, out_channels, norm=None, lateral_activation=None, output_activation=F.relu, top_block=None, fuse_type="sum", out_as_input=False):
        """
        Arguments:
            out_channels (int): number of channels in the output feature maps.
            norm (nn.BatchNorm2d or None): the normalization to use.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                FPN output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra FPN levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            fuse_type (str): types for fusing the top down features and the lateral
                ones. It can be "sum" (default), which sums up element-wise; or "avg",
                which takes the element-wise mean of the two.
        """
        super(FPN, self).__init__()

        # Feature map strides and channels from the bottom up network (e.g. ResNet)
        input_channels_stride = input_channels_stride
        strides = [f[1] for f in input_channels_stride]
        in_channels_per_feature = [f[0] for f in input_channels_stride]

        _assert_strides_are_log2_contiguous(strides)
        lateral_convs = []
        output_convs = []
        upsamples = []
        for idx, in_channels in enumerate(in_channels_per_feature):
            if norm is not None:
                lateral_norm = output_norm = norm
            else:
                lateral_norm = output_norm = None

            lateral_conv = Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, norm=lateral_norm, activation=lateral_activation
            )

            if out_as_input:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='bilinear'),
                    Conv2d(out_channels, out_channels, kernel_size=1, bias=True)
                )

                output_conv = Conv2d(
                     out_channels,
                     in_channels,
                     kernel_size=1,
                     stride=1,
                     padding=0,
                     activation=F.gelu,
                )
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2.0, mode='nearest'),
                )
                output_conv = Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm=output_norm,
                    activation=output_activation,
                )

            stage = int(math.log2(strides[idx]))
            upsamples.append(upsample)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.upsamples = upsamples[::-1]
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.upsamples = nn.ModuleList(self.upsamples)
        self.lateral_convs = nn.ModuleList(self.lateral_convs)
        self.output_convs = nn.ModuleList(self.output_convs)

        self.top_block = top_block
        self.out_channels = out_channels
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {"p{}".format(int(math.log2(s))): s for s in strides}
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        assert fuse_type in {"avg", "sum"}
        self._fuse_type = fuse_type

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=1.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    @property
    def size_divisibility(self):
        return self._size_divisibility

    def forward(self, backbone_features):
        """
        Arguments:
            backbone_features (dict[str->Tensor]): mapping feature map name (e.g., "res5") to
                feature map tensor for each feature level in high to low resolution order.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to FPN feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                paper convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        for idx, feature in enumerate(backbone_features):
            b = feature.size(0)
            h = w = int(feature.size(1) ** .5)
            backbone_features[idx] = backbone_features[idx].view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()

        results = []
        prev_features = self.lateral_convs[0](backbone_features[0])
        results.append(self.output_convs[0](prev_features))  # .flatten(2).permute(0,2,1)

        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs, self.output_convs)):
            # Slicing of ModuleList is not supported https://github.com/pytorch/pytorch/issues/47336
            # Therefore we loop over all modules but skip the first one
            if idx > 0:
                features = backbone_features[idx]
                top_down_features = self.upsamples[idx](prev_features)
                lateral_features = lateral_conv(features)
                prev_features = lateral_features + top_down_features
                if self._fuse_type == "avg":
                    prev_features /= 2
                results.append(output_conv(prev_features))

        if self.top_block is not None:
            if self.top_block.in_feature in backbone_features:
                top_block_in_feature = backbone_features[self.top_block.in_feature]
            else:
                top_block_in_feature = results[self._out_features.index(self.top_block.in_feature)]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        # return {f: res for f, res in zip(self._out_features, results)}
        return results

    def extra_repr(self) -> str:
        return "".join([f'{k}: {v}\n' for k, v in self.__dict__.items() if k != '_modules'])


def _assert_strides_are_log2_contiguous(strides):
    """
    Assert that each stride is 2x times its preceding stride, i.e. "contiguous in log2".
    """
    for i, stride in enumerate(strides[1:], 1):
        assert stride == 2 * strides[i - 1], "Strides {} {} are not log2 contiguous".format(
            stride, strides[i - 1]
        )


class LastLevelMaxPool(nn.Module):
    """
    This module is used in the original FPN to generate a downsampled
    P6 feature from P5.
    """

    def __init__(self):
        super().__init__()
        self.num_levels = 1
        self.in_feature = "p5"

    def forward(self, x):
        return [F.max_pool2d(x, kernel_size=1, stride=2, padding=0)]


class LastLevelP6P7(nn.Module):
    """
    This module is used in RetinaNet to generate extra layers, P6 and P7 from
    C5 feature.
    """

    def __init__(self, in_channels, out_channels, in_feature="res5"):
        super().__init__()
        self.num_levels = 2
        self.in_feature = in_feature
        self.p6 = Conv2d(in_channels, out_channels, 3, 2, 1)
        self.p7 = Conv2d(out_channels, out_channels, 3, 2, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, a=1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, c5):
        p6 = self.p6(c5)
        p7 = self.p7(F.relu(p6))
        return [p6, p7]


