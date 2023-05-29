import torch
from torch.nn import functional as F

from model.decoders import SwinExpand, SwinSegformer, SwinFPN, SwinFPNUperNet
from model.fpn import FPN
from model.swin import SwinEncoder
from model.swin_unet import SwinUnet


def load_pretrained(path, model, dont_load_keywords=(), verbose=False):

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index, relative_coords_table and attn_mask since we always re-init it
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    state_dict_2 = {}
    for k, i in state_dict.items():
        if k.startswith('patch_embed') or k.startswith('absolute_pos_embed'):
            state_dict_2[k] = i
        elif 'head' in k:
            continue
        else:
            state_dict_2[f"encoder.{k}"] = i

    state_dict = state_dict_2
    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict().get(k, None)
        if relative_position_bias_table_current is None:
            continue
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()

        if nH1 != nH2:
            print(f"Error in loading {k}, passing......")
            state_dict.pop(k)
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = F.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2), mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            print(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = F.interpolate(absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    if not model.return_inter:
        torch.nn.init.constant_(model.encoder.head.bias, 0.)
        torch.nn.init.constant_(model.encoder.head.weight, 0.)

    state_dict = {k: v for k, v in state_dict.items() if not any(word in k for word in dont_load_keywords)}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(missing_keys)
        print(unexpected_keys)

        removed = ['relative_position_index', 'relative_coords_table', 'attn_mask']
        weird = list(filter(lambda k: not any(s in k for s in removed), missing_keys))
        print(f"WEIRD: {weird}")

    del checkpoint
    torch.cuda.empty_cache()
    print(f"Loaded model from {path}")


def build_models(im_size, use_fpn, decoder_name, head_name, model_config, num_classes, extra_losses, verbose=False):

    model = {
        'tiny': {
            'embed': 96,
            'depths': [2, 2, 6, 2],
            'num_heads': [3, 6, 12, 24],
            'checkpoint': 'pretrained/swin_t_22k.pth'
        },
        'small': {
            'embed': 96,
            'depths': [2, 2, 18, 2],
            'num_heads': [3, 6, 12, 24],
            'checkpoint': 'pretrained/swin_s_22k.pth'
        }
    }

    sel_model = model.get(model_config, None)

    # special case for swin-unet training
    if sel_model is None:
        if model_config == 'swin-unet':
            net = SwinUnet(num_classes=num_classes)
            net.load_from("pretrained/swin_t_22k.pth")

            from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
            flop = FlopCountAnalysis(net, torch.randn(8, 3, 224, 224))
            print(flop_count_table(flop, max_depth=4))
            return net

    intermediate = decoder_name is not None

    model = SwinEncoder(
        img_size=im_size,
        patch_size=4,
        embed_dim=sel_model['embed'],
        depths=sel_model['depths'],
        num_heads=sel_model['num_heads'],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        norm_layer=torch.nn.LayerNorm,
        ape=False,
        patch_norm=True,
        num_classes=num_classes,
        return_intermediate=intermediate,
    )

    model = build_decoder(model, use_fpn, num_classes, decoder_name=decoder_name, head_name=head_name)
    if decoder_name is not None:
        load_pretrained(sel_model['checkpoint'], model.backbone, verbose=verbose)
    else:
        load_pretrained(sel_model['checkpoint'], model, verbose=verbose)

    if 'cls' not in extra_losses:
        model.backbone.reset_head()

    if verbose:
        from fvcore.nn import FlopCountAnalysis, flop_count_table, flop_count_str
        flop = FlopCountAnalysis(model, torch.randn(8, 3, 224, 224))
        print(flop_count_table(flop, max_depth=4))
        print(flop_count_str(flop))
        print(flop.total())

    return model


def get_fpn(in_channels, out_channels, input_as_output=False):
    fpn = FPN(
        input_channels_stride=in_channels,
        out_channels=out_channels,
        norm=None,
        lateral_activation=None,
        top_block=None,
        fuse_type='sum',
        out_as_input=input_as_output

    )
    return fpn


def build_decoder(backbone, add_fpn, num_classes, decoder_name, head_name):
    out_channels = 256

    if not add_fpn and decoder_name is None:
        return backbone

    if add_fpn:
        if decoder_name == 'patch_expand':
            fpn = get_fpn(backbone._out_features, out_channels, input_as_output=True)
        else:
            fpn = get_fpn(backbone._out_features, out_channels)
    else:
        fpn = None

    if decoder_name == 'swin_fpn':
        z = SwinFPN(backbone, fpn, num_classes, head_name=head_name)
    elif decoder_name == 'patch_expand':
        z = SwinExpand(backbone, fpn, num_classes, head_name=head_name)
    elif decoder_name == 'segformer':
        z = SwinSegformer(backbone, fpn, num_classes, head_name=head_name)
    elif decoder_name == 'upernet':
        z = SwinFPNUperNet(backbone, fpn, num_classes, head_name=head_name)
    else:
        raise NotImplemented("Decoder not valid")

    return z


