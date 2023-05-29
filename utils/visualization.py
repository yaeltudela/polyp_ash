import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from kornia import augmentation as kaug
from torch import nn
from torch.nn import functional as TF
from torchvision.utils import make_grid, save_image

from utils.utils import resize

denormalize = kaug.Denormalize(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225]),
                               keepdim=True)


def show_batch(images, labels, masks=None):
    if masks is not None:
        f, axes = plt.subplots(1, 2)
    else:
        f, axes = plt.subplots(1)
        masks = [None] * images.size(0)

    if labels is None:
        labels = [None] * images.size(0)

    images = images.cpu()
    if masks is not None:
        masks = masks.cpu()

    for idx, (im, l, mask) in enumerate(zip(images, labels, masks)):
        if mask is None:
            axes[idx].set_title(l)
            axes[idx].imshow((denormalize(im).squeeze().permute(1, 2, 0) * 255).int())

        else:
            axes[0].set_title(l)
            axes[0].imshow((denormalize(im).squeeze().permute(1, 2, 0) * 255).int())
            axes[1].imshow(mask.argmax(0).int())

        f.tight_layout()
        f.show()


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, dpi=300)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()
    plt.show()


def sample_transforms(images, transform_function, masks=None):
    inputs = denormalize(transform_function(images, masks))
    images = denormalize(images)

    view = []
    if masks is None:
        masks = [None] * len(images)

    for im, m, out in zip(images, masks, inputs):
        if m is None:
            m = torch.zeros(3, im.size(-2), im.size(-1))
        else:
            m = TF.interpolate(m[None, None].float(), size=(im.size(-2), im.size(-1))).squeeze(0)
            m = (m.repeat(3, 1, 1)).int()

        view += [im, m, out]

    view = make_grid(view, nrow=3)

    show(view)


gt_embedding = nn.Embedding.from_pretrained(torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])).cuda()


@torch.no_grad()
def save_model_results(images, ground_truth, predictions, save_to_disk):
    threshold = 0.
    b, c, h, w = images.shape
    images = denormalize(images)
    embeding_classes = nn.Embedding.from_pretrained(
        torch.FloatTensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])).cuda()

    if ground_truth.ndim < 4:
        ground_truth = TF.one_hot(ground_truth).permute(0, 3, 1, 2)

    predictions = resize(predictions, (h, w))
    predictions = predictions.sigmoid()
    if threshold > 0.:
        predictions[predictions < threshold] = 0.
    predictions = embeding_classes(predictions.argmax(1)).permute(0, 3, 1, 2)

    ground_truth = ground_truth[:, 1:]
    ground_truth = gt_embedding(ground_truth.argmax(1)).permute(0, 3, 1, 2) * ground_truth
    ground_truth = resize(ground_truth, (h, w), mode='nearest')

    view = []
    for idx, (im, gt, out) in enumerate(zip(images, ground_truth, predictions)):
        # to see anything, must remove if preds could contain negative frames
        no_pred = out.sum() == 0
        if no_pred:
            out = embeding_classes((predictions[idx][1:].argmax(0) + 1)).permute(2, 0, 1)
            out[:, :20, :20] = 1

        opacity = 0.4
        gt = (im * (1 - opacity) + gt * opacity)
        out = (im * (1 - opacity) + out * opacity)

        view += [im, gt, out, ]

    if save_to_disk:
        os.makedirs("/".join(save_to_disk.split("/")[:-1]), exist_ok=True)

        save_image(view, fp=save_to_disk, nrow=6)
    else:
        view = make_grid()
        show(view)
