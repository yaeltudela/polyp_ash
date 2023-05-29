import warnings
from typing import Optional

import torch
from einops import reduce
from kornia.utils.one_hot import one_hot
from torch import nn
from torch.nn import functional as F
from torch.nn.functional import logsigmoid


class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, dim=-1):
        super(SoftTargetCrossEntropy, self).__init__()
        self.dim = dim
        self.ignore_index = None

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(x, dim=self.dim)

        loss = torch.sum(-target * log_probs, dim=self.dim)
        return loss.mean()


# Cross Entropy with weight per sample (preds: b c h w, target: b c h w)
class FocalLoss(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Arguments:
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stability. This is no longer
          used.

    Shape:
        - Input: :math:`(N, C, *)` where C = number of classes.
        - Target: :math:`(N, *)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    """

    def __init__(self, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None,
                 label_smoothing=0.) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.eps: Optional[float] = eps
        self.label_smooth = label_smoothing

    def forward(self, preds, target):
        return focal_loss(preds, target, self.alpha, self.gamma, self.reduction, self.eps, self.label_smooth)


# Cross Entropy that adds smoothing (preds: b c h w, target: b c h w)
class CrossEntropyLossWithLabelSmoothing(nn.Module):
    __constants__ = ["reduction", "ignore_index", "smooth_factor"]

    def __init__(
            self,
            reduction: str = "mean",
            label_smoothing=None,
            ignore_index=None,
            dim: int = 1,
            weight=None,
    ):

        super().__init__()
        self.smooth_factor = label_smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.dim = dim
        self.weights = weight
        if self.weights is not None:
            self.weights = self.weights.cuda()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        log_prob = F.log_softmax(y_pred, dim=self.dim)
        y_true = y_true.long()

        if self.weights is not None:
            weights = self.weights[y_true]
            if self.dim == -1:
                weights.unsqueeze_(1)

            log_prob = weights * log_prob

        return label_smoothed_nll_loss(
            log_prob,
            y_true,
            smooth_factor=self.smooth_factor,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            dim=self.dim,
        )


class TverskyLoss(nn.Module):
    """Tversky loss for image segmentation task.
    Where FP and FN is weighted by alpha and beta params.
    With alpha == beta == 0.5, this loss becomes equal DiceLoss.
    It supports binary, multiclass and multilabel cases
    Arguments:
        mode: Metric mode {'binary', 'multiclass', 'multilabel'}
        log_loss: If True, loss computed as ``-log(tversky)`` otherwise ``1 - tversky``
        from_logits: If True assumes input is raw logits
        smooth:
        eps: Small epsilon for numerical stability
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
    Return:
        loss: torch.Tensor
    """

    def __init__(self, mode: str, log_loss=False, from_logits=True, ignore_index=-100, smooth=0.0, eps=1e-7, alpha=0.5,
                 beta=0.5, gamma=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.ignore_index = ignore_index if ignore_index >= 0 else None

        self.mode = mode

        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss
        if alpha == beta:
            if alpha == .5:
                print("Tversky index as Dice Loss")
            if alpha == 1.:
                print("Tversky index as Jaccard Loss")

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == 'multi':
                y_pred = y_pred.log_softmax(dim=1).exp()
            else:
                y_pred = F.logsigmoid(y_pred).exp()

        b, num_classes = y_true.size(0), y_pred.size(1)
        dims = (1, 2)

        if self.mode == 'binary':
            y_true = y_true.view(b, 1, -1)
            y_pred = y_pred.view(b, 1, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        if self.mode == 'multi':
            y_true = y_true.view(b, num_classes, -1)
            y_pred = y_pred.view(b, num_classes, -1)

            if self.ignore_index is not None:
                mask = y_true != self.ignore_index
                y_pred = y_pred * mask
                y_true = y_true * mask

        scores = soft_tversky_score(y_pred, y_true.type_as(y_pred), self.alpha, self.beta, self.smooth, self.eps, dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss

        mask = (y_true.sum(dims) > 0).float()
        loss *= mask

        loss = torch.pow(loss, self.gamma)

        return loss.mean()


def dice_score(y_pred, y_true, reduction='mean'):
    scores = soft_tversky_score(y_pred, y_true, alpha=0.5, beta=0.5, smooth=0, dims=(2, 3))

    if reduction == 'mean':
        scores = scores.mean()

    return scores


def focal_loss(preds, target, alpha: float, gamma: float = 2.0, reduction: str = 'none', eps: Optional[float] = None,
               smooth_factor: Optional[float] = 0.0):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Arguments:
        preds: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Return:
        the computed loss.
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    # compute softmax over the classes axis
    input_soft = preds.softmax(1)
    log_input_soft = preds.log_softmax(1)

    # create the labels one hot tensor
    if target.ndim != 4 and not target.is_floating_point():
        target_one_hot = one_hot(target, num_classes=preds.shape[1], device=preds.device, dtype=preds.dtype)
    else:
        target_one_hot = target

    eps = smooth_factor / (target_one_hot.size(1) - 1)
    target_one_hot = target_one_hot * eps + (1 - target_one_hot) * (1 - eps)

    # compute the actual focal loss (difficulty level)
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


def label_smoothed_nll_loss(log_prob: torch.Tensor, target: torch.Tensor, smooth_factor: float, ignore_index=None,
                            reduction="mean", dim=-1, **kwargs) -> torch.Tensor:
    """
    Source: https://github.com/pytorch/fairseq/blob/master/fairseq/criterions/label_smoothed_cross_entropy.py
    :param log_prob: Log-probabilities of predictions (e.g. after log_softmax)
    :param target:
    :param smooth_factor:
    :param ignore_index:
    :param reduction:
    :return:
    """
    if target.dim() == log_prob.dim() - 1:
        target = target.unsqueeze(dim)

    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        target = target.masked_fill(pad_mask, 0)
        nll_loss = -log_prob.gather(dim=dim, index=target)
        smooth_loss = -log_prob.sum(dim=dim, keepdim=True)

        # nll_loss.masked_fill_(pad_mask, 0.0)
        # smooth_loss.masked_fill_(pad_mask, 0.0)
        nll_loss = nll_loss.masked_fill(pad_mask, 0.0)
        smooth_loss = smooth_loss.masked_fill(pad_mask, 0.0)
    else:
        nll_loss = -log_prob.gather(dim=dim, index=target)
        smooth_loss = -log_prob.sum(dim=dim, keepdim=True)

        nll_loss = nll_loss.squeeze(dim)
        smooth_loss = smooth_loss.squeeze(dim)

    if reduction == "sum":
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    if reduction == "mean":
        nll_loss = nll_loss.mean()
        smooth_loss = smooth_loss.mean()

    eps_i = smooth_factor / log_prob.size(dim)
    loss = (1.0 - smooth_factor) * nll_loss + eps_i * smooth_loss
    return loss


def soft_tversky_score(output: torch.Tensor, target: torch.Tensor, alpha: float, beta: float, smooth: float = 0.0,
                       eps: float = 1e-7, dims=None, ) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)  # TP
        fp = torch.sum(output * (1.0 - target), dim=dims)
        fn = torch.sum((1 - output) * target, dim=dims)
    else:
        intersection = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1.0 - target))
        fn = torch.sum((1 - output) * target)

    tversky_score = (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)
    return tversky_score


class MultiLabelSoftMarginLoss(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return soft_margin_ranking_loss(y_pred, y_true, self.reduction)


def soft_margin_ranking_loss(outputs, target, reduction='none'):
    loss = -(target * logsigmoid(outputs) + (1 - target) * logsigmoid(-outputs))

    # if weight is not None:
    #     loss = loss * weight
    loss = reduce(loss, 'b c h w -> b', reduction='mean')

    if reduction == "none":
        ret = loss
    elif reduction == "mean":
        ret = loss.mean()
    elif reduction == "sum":
        ret = loss.sum()
    else:
        raise ValueError(reduction + " is not valid")
    return ret
