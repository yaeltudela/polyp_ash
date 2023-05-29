import math
import random
from typing import Union, Iterable, Tuple

import albumentations as A
from albumentations.augmentations.crops import functional as AF
import cv2
import numpy as np
import torch
from torch.nn import functional as TF


def _autocontrast(img, **params):
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    h = cv2.calcHist([img], [0], None, [256], (0, 256)).ravel()

    for lo in range(256):
        if h[lo]:
            break
    for hi in range(255, -1, -1):
        if h[hi]:
            break

    if hi > lo:
        lut = np.zeros(256, dtype=np.uint8)
        scale_coef = 255.0 / (hi - lo)
        offset = -lo * scale_coef
        for ix in range(256):
            lut[ix] = int(np.clip(ix * scale_coef + offset, 0, 255))

        img = cv2.LUT(img, lut)

    return img


def one_hot_annots(num_classes, p=1.):

    def mask_to_one_hot(mask, **params):
        return TF.one_hot(mask.long(), num_classes).permute(2, 0, 1)

    return A.Lambda(name='OneHot', mask=mask_to_one_hot, p=p)


def auto_contrast(p=0.5):
    return A.Lambda(name='AutoContrast', image=_autocontrast, p=p)


def _color_enhance(img, **params):
    factor = random.gauss(10, 1) / 10 * 1.8 + 0.1
    degenerate = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    degenerate = cv2.cvtColor(degenerate, cv2.COLOR_GRAY2BGR)
    return img * (1 - factor) + degenerate * factor


def color_enhance(p=.5):
    return A.Lambda(name='Color', image=_color_enhance, p=p)


class CutOut(A.DualTransform):

    def __init__(self, im_size, min_holes, max_holes,  min_area=0.02, max_area=1/3, min_aspect=0.3, max_aspect=None,
                 num_splits=0, bg_class_id=None, p: float = 0.5, fill_mode='pixel'):
        super().__init__(always_apply=False, p=p)
        self.im_size = im_size

        self.probability = p
        self.min_area = min_area
        self.max_area = max_area
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
        self.min_holes = min_holes
        self.max_holes = max_holes or min_holes
        self.num_splits = num_splits

        self.fill_mode = fill_mode
        self.bg_class_id = bg_class_id

    def get_params(self):
        holes = []
        img_h, img_w = self.im_size
        area = img_w * img_h
        count = self.min_holes if self.min_holes == self.max_holes else random.randint(self.min_holes, self.max_holes)
        for _ in range(count):
            for attempt in range(10):

                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)

                    holes.append((top, h, left, w))

                    break
        return {
            'holes': holes
        }

    def apply(self, img: np.ndarray, fill_value: Union[int, float] = 0, holes: Iterable[Tuple[int, int, int, int]] = (),
              **params) -> np.ndarray:
        return self.cutout(img, holes)

    def apply_to_mask(self, img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]] = (), **params) -> np.ndarray:
        if self.bg_class_id is None:
            return img
        return self.cutout(img, holes, self.bg_class_id)

    def cutout(self, img: np.ndarray, holes: Iterable[Tuple[int, int, int, int]], bg_class_id=None) -> np.ndarray:
        # for x1, y1, x2, y2 in holes:
        for top, h, left, w in holes:
            if bg_class_id is not None:
                # patch_size = (h, w)
                # continue
                img[top:top+h, left:left+w] = bg_class_id
            else:
                patch_size = (img.size(0), h, w)
                img[:, top:top+h, left:left+w] = self._get_pixels(patch_size, fill_value=bg_class_id)

        return img

    def _get_pixels(self, patch_size, dtype=torch.float32, device='cpu', fill_value=None):
        if fill_value is not None:
            return torch.ones(patch_size, dtype=dtype, device=device) * fill_value

        if self.fill_mode == 'pixel':
            return torch.empty(patch_size, dtype=dtype, device=device).normal_()
        elif self.fill_mode == 'color':
            return torch.empty((patch_size[0], 1, 1), dtype=dtype, device=device).normal_()
        else:
            return torch.zeros((patch_size[0], 1, 1), dtype=dtype, device=device)

    def get_transform_init_args_names(self):
        return (
            "im_size",
            "min_area",
            "max_area",
            "log_aspect_ratio",
            "min_holes",
            "max_holes",
            "num_splits",
            "fill_mode",
            "bg_class_id",
        )


class CatRatioRandomResizedCrop(A.RandomResizedCrop):

    def __init__(self, height, width, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333),
                 interpolation=cv2.INTER_LINEAR, crop_max_ratio=1., always_apply=False, p=1.0):
        super().__init__(height, width, scale, ratio, interpolation, always_apply, p)
        self.crop_max_ratio = crop_max_ratio

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        mask = params['mask']
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                h_start = i * 1.0 / (img.shape[0] - h + 1e-10)
                w_start = j * 1.0 / (img.shape[1] - w + 1e-10)

                tmp_mask = AF.random_crop(img, h, w, h_start, w_start)

                labels, cnt = np.unique(tmp_mask, return_counts=True)
                if len(cnt) > 1 and cnt.max() / cnt.sum() < self.crop_max_ratio:
                    return {
                        "crop_height": h,
                        "crop_width": w,
                        "h_start": h_start,
                        "w_start": w_start,
                    }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    @property
    def targets_as_params(self):
        return ['image', 'mask']

    def get_transform_init_args_names(self):
        return *super(CatRatioRandomResizedCrop, self).get_transform_init_args_names(), "crop_max_ratio"
