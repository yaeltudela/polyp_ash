import numpy as np
import torch
from sklearn.metrics import recall_score, precision_score, fbeta_score, matthews_corrcoef, \
    confusion_matrix, accuracy_score


def specificity(cf, tn=None, fp=None):
    if tn is not None and fp is not None:
        return tn / (fp + tn)
    if cf is None:
        raise NotImplemented('CF or metrics should be not None')
    labels = np.arange(cf.shape[0])
    spec_per_class = np.zeros_like(labels, dtype=float)
    for label in labels:
        tp, fp, tn, fn = cf_per_class(cf, label)

        spec_per_class[label] = tn / (fp + tn)

    return spec_per_class


def cf_per_class(cf, label):
    tp = cf[label, label]
    fp = cf[:, label].sum() - tp
    fn = cf[label, :].sum() - tp
    tn = cf.sum() - tp - fp - fn
    return tp, fp, tn, fn


def negative_predictive_value(cf=None, tn=None, fn=None):
    if tn is not None and fn is not None:
        return tn / (fn + tn)
    if cf is None:
        raise NotImplemented('CF or metrics should be not None')
    labels = np.arange(cf.shape[0])
    npv_per_class = np.zeros_like(labels, dtype=float)
    for label in labels:
        tp, fp, tn, fn = cf_per_class(cf, label)

        npv_per_class[label] = tn / (fn + tn)

    return npv_per_class


def iou_score(output, targets, reduction='mean'):
    output, targets = output.cpu(), targets.cpu()

    targets = targets.argmax(1)
    b, num_classes, _, _ = output.shape
    _, output = torch.max(output, dim=1)
    # create a tensor to store IoU for each class
    ious = torch.zeros(num_classes)
    for c in range(num_classes):
        pred_idx = output == c
        label_idx = targets == c
        intersection = (pred_idx & label_idx).sum().float()
        union = (pred_idx | label_idx).sum().float()
        if union != 0:
            ious[c] = intersection / union
    if reduction == 'mean':
        return ious.mean()
    else:
        return ious


def compute_cls_metrics(acc1_meter, loss_meter, preds, targets, num_classes):
    avg = 'binary' if num_classes == 2 else None
    if acc1_meter is None:
        acc = accuracy_score(targets, preds)
    else:
        acc = acc1_meter.avg

    recall = recall_score(targets, preds, average=avg, pos_label=0)
    precision = precision_score(targets, preds, average=avg, pos_label=0)
    f1 = fbeta_score(targets, preds, beta=1, average=avg, pos_label=0)
    f2 = fbeta_score(targets, preds, beta=2, average=avg, pos_label=0)
    mcc = matthews_corrcoef(targets, preds)
    cf = confusion_matrix(targets, preds)
    if avg == 'binary':
        targets = (~targets.bool()).int()
        preds = (~preds.bool()).int()
        tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
        npv = negative_predictive_value(None, tn, fn)
        spec = specificity(None, tn, fp)
    else:
        npv = negative_predictive_value(cf)
        spec = specificity(cf)
    results = [
        f' * Acc@1 {acc:.3f}',
        f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t' if loss_meter is not None else '',
        cf,
        f'Precision: {precision} - mean: {precision.mean():.4f}',
        f'Recall: {recall} - mean: {recall.mean():.4f}',
        f'NPV: {npv}- mean: {npv.mean():.4f}',
        f'Specificity: {spec} - mean: {spec.mean():.4f}',
        f'F1 score: {f1} - mean: {f1.mean():.4f}',
        f'F2 score: {f2} - mean: {f2.mean():.4f}',
        f"MCC : {mcc}"
    ]
    return cf, f1, f2, mcc, npv, precision, recall, results, spec
