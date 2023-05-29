import torch
from torch import nn

from losses import FocalLoss, SoftTargetCrossEntropy, CrossEntropyLossWithLabelSmoothing, TverskyLoss, \
    MultiLabelSoftMarginLoss


def get_criterion_params(name, label_smoothing, **kwargs):
    if name == 'cross_entropy':
        params = {'weight': kwargs.get('cls_weight', None)}
    elif name == 'soft_ranking':
        params = {'weight': kwargs.get('cls_weight', None)}
    elif name == 'focal':
        params = {
            'alpha': 1.,
            'gamma': 2.,
            'reduction': 'mean',
            'label_smoothing': label_smoothing
        }
    elif name == 'label_smooth_cross_entropy':
        params = {'label_smoothing': label_smoothing, 'dim': 1, 'weight': kwargs.get('cls_weight', None)}
    elif name == 'soft_label_cross_entropy':
        params = {'dim': 1, 'weight': kwargs.get('cls_weight', None)}
    else:
        raise NotImplemented(f"{name} is not implemented")

    return params


def build_criterion(mixup_fn, label_smoothing, is_focal=False, name=None, **kwargs):
    if name is None:
        # smoothing is handled with mix_up label transform
        if mixup_fn is not None:
            name = 'soft_label_cross_entropy'
        # no mix_up but label smoothing
        elif label_smoothing > 0.:
            name = 'label_smooth_cross_entropy'
        # no mix_up and no label smoothing
        else:
            name = 'cross_entropy'
        if is_focal:
            name = 'focal'

    params = get_criterion_params(name, label_smoothing, **kwargs)
    criterion = build_criterion_by_name(name, **params)

    print(criterion)
    return criterion


def build_criterion_by_name(name, **kwargs):
    if name == 'cross_entropy':
        return torch.nn.CrossEntropyLoss(**kwargs)
    elif name == 'soft_ranking':
        return MultiLabelSoftMarginLoss(**kwargs)
    elif name == 'soft_label_cross_entropy':
        return SoftTargetCrossEntropy(**kwargs)
    elif name == 'label_smooth_cross_entropy':
        return CrossEntropyLossWithLabelSmoothing(**kwargs)
    else:
        raise NotImplemented("Loss not valid")


val_criterion = torch.nn.CrossEntropyLoss()


class Criterion(nn.Module):
    def __init__(self, name, mixup_fn, label_smoothing, is_focal=False, smooth_tvesky=0., extra_losses=(),
                 is_train=True, **kwargs):
        super().__init__()
        self.name = name
        if is_train:
            self.criterion = build_criterion(mixup_fn, label_smoothing, is_focal, name, **kwargs)
        else:
            self.criterion = build_criterion(None, 0., is_focal, name, **kwargs)

        ignore_index = kwargs.get('ignore_index', -100)

        self.extra_losses = {}
        for loss in extra_losses:
            if loss == 'iou':
                if is_train:
                    alpha = 0.6
                    extra = TverskyLoss('multi', log_loss=False, from_logits=True, alpha=alpha, beta=(1. - alpha),
                                        smooth=smooth_tvesky, gamma=2., ignore_index=ignore_index)
                else:
                    extra = TverskyLoss('multi', log_loss=False, from_logits=True, alpha=.5, beta=.5, smooth=0,
                                        ignore_index=ignore_index)
            elif loss == 'cls':
                if is_train:
                    extra = FocalLoss(alpha=1., reduction='mean') if is_focal else nn.CrossEntropyLoss(
                        ignore_index=ignore_index)
                else:
                    extra = nn.CrossEntropyLoss()
            else:
                raise NotImplemented('loss not implemented')
            self.extra_losses[loss] = extra

        self.loss_weights = {
            'cls': 1.,
            'iou': 1.,
        }

    def forward(self, losses_dict):
        losses = {}

        base_losses = [loss for loss in losses_dict.keys() if 'base' in loss]
        for name in base_losses:
            values = losses_dict[name]
            losses[name] = self.criterion(values['preds'], values['targets'])

            # factor = values.get('scale_factor', None)
            # if factor:
            #     losses[name] = losses[name] * factor

        for name, criterion in self.extra_losses.items():
            values = losses_dict[name]
            losses[name] = self.extra_losses[name](values['preds'], values['targets'])

        all_loss = 0
        for loss, value in losses.items():
            all_loss += value * self.loss_weights.get(loss, 1.0)

        return all_loss

    def __str__(self):
        return f"Criterion:\n - Base: {self.name}\n - Extra losses: " \
               f"{[f'    - {k}: {v.__dict__}    ' for (k, v) in self.extra_losses.items()]} " \
               f"\n weights: {self.loss_weights}"
