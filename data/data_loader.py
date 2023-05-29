import albumentations as A
import cv2
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, WeightedRandomSampler

from data.polyp import PolypCls, PolypSegm
from data.transforms import CutOut, auto_contrast, color_enhance, CatRatioRandomResizedCrop, one_hot_annots
from utils.utils import seed_everything


def is_debug():
    from sys import gettrace
    return gettrace() is not None


def seed_worker(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_everything(initial_seed + worker_id)


def get_train_test_loaders(im_size, batch_size, dataset_name, num_workers, use_ass, only_polyp=False,
                           generator_seed=873):
    num_workers = num_workers if not is_debug() else 0
    prefetch = 4 if num_workers != 0 else 2

    num_classes, test_ds, train_ds, val_ds = build_dataset(dataset_name, im_size, use_ass, only_polyp)

    print(train_ds)
    print(test_ds)
    if 'polyp_ash' in dataset_name:
        weights_sampler = train_ds.compute_sample_weights()

        train_sampler = WeightedRandomSampler(weights_sampler, len(train_ds))
    else:
        train_sampler = None

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True if train_sampler is None else False,
                              num_workers=num_workers, pin_memory=True, prefetch_factor=prefetch, sampler=train_sampler,
                              drop_last=True, worker_init_fn=seed_worker, generator=generator_seed)
    if val_ds is not None:
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    else:
        val_loader = None
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return (train_loader, val_loader, test_loader), num_classes


def build_transforms(im_size, num_classes):
    scale = (.4, 1.3)
    ratio = (1 / 2, 3 / 2)
    # crop_pct = 0.9
    # scale_size = int(math.floor(im_size / crop_pct))
    affine = 25
    rel_translate = .45
    bright_contrast = .5

    transform_x = A.Compose([
        CatRatioRandomResizedCrop(im_size, im_size, scale=scale, ratio=ratio, crop_max_ratio=0.65),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.OneOf([
            A.Defocus(),
            A.MotionBlur(),
            A.ZoomBlur()
        ],
            p=0.1
        ),
        A.SomeOf(
            [
                A.Rotate((-30, 30), border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3),
                A.Affine(shear={'x': (-affine, affine), 'y': (0, 0)}, mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0,
                         p=0.2),
                A.Affine(shear={'x': (0, 0), 'y': (-affine, affine)}, mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0,
                         p=0.2),
                A.Affine(translate_percent={'x': (-rel_translate, rel_translate), 'y': (0, 0)},
                         mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0,
                         p=0.1),
                A.Affine(translate_percent={'x': (0, 0), 'y': (-rel_translate, rel_translate)},
                         mode=cv2.BORDER_CONSTANT, cval=0, cval_mask=0,
                         p=0.1),
                color_enhance(p=0.025),
                A.Sharpen((0, 0.5), p=0.025),
                auto_contrast(p=0.025),
                A.Solarize(p=0.01),
                A.RandomBrightnessContrast(brightness_limit=bright_contrast, contrast_limit=0.0, p=0.005),
                A.RandomBrightnessContrast(brightness_limit=0.0, contrast_limit=bright_contrast, p=0.005),
                A.Equalize(mode='pil', p=0.005),

            ],
            n=2, replace=False
        ),
        A.Normalize(),
        ToTensorV2(transpose_mask=True),
        CutOut(im_size=(im_size, im_size), min_holes=1, max_holes=2, p=0.25, bg_class_id=0),
        one_hot_annots(num_classes=num_classes),

    ])
    transform_x_test = A.Compose([
        A.Resize(im_size, im_size, always_apply=True),

        A.Normalize(always_apply=True),
        ToTensorV2(always_apply=True, transpose_mask=True),
        one_hot_annots(num_classes=num_classes),

    ])
    return transform_x, transform_x_test


def build_dataset(dataset_name, im_size, use_ass, only_polyp=False):
    num_classes_dataset = {
        'CIFAR10': 10,
        'CIFAR100': 100,
        'polyp_ash': 3 if use_ass else (1 if only_polyp else 2),
        'VOC': 20,
        'KVASIR': 1,
        'CLINIC': 1,
        'COLON': 1,
    }

    num_classes = num_classes_dataset.get(dataset_name, 1)
    transform_x, transform_x_test = build_transforms(im_size, num_classes=num_classes + 1)

    if dataset_name == 'polyp_ash':
        train_ds = PolypCls("datasets/challenge_cls", 'train.csv', split='train', use_ass=use_ass,
                            only_polyp=only_polyp,
                            transforms=transform_x)
        val_ds = PolypCls("datasets/challenge_cls", 'valid.csv', split='valid', use_ass=use_ass, only_polyp=only_polyp,
                          transforms=transform_x_test)
        test_ds = PolypCls("datasets/challenge_cls", 'test.csv', split='test', use_ass=use_ass, only_polyp=only_polyp,
                           transforms=transform_x_test)

    elif dataset_name == 'test_CLINIC':
        train_ds = val_ds = None
        test_ds = PolypSegm("datasets/TestDataset/CVC-ClinicDB", annot_extension='png', im_extension='png',
                            transforms=transform_x_test)
    elif dataset_name == 'test_COLON':
        train_ds = val_ds = None
        test_ds = PolypSegm("datasets/TestDataset/CVC-ColonDB", annot_extension='png', im_extension='png',
                            transforms=transform_x_test)
    elif dataset_name == 'test_ETIS':
        train_ds = val_ds = None
        test_ds = PolypSegm("datasets/TestDataset/ETIS-LaribPolypDB", annot_extension='png', im_extension='png',
                            transforms=transform_x_test)
    elif dataset_name == 'test_KVASIR':
        train_ds = val_ds = None
        test_ds = PolypSegm("datasets/TestDataset/Kvasir", annot_extension='png', im_extension='png',
                            transforms=transform_x_test)
    elif dataset_name == 'split_train':
        train_ds = PolypSegm('datasets/TrainDataset', annot_extension='png', im_extension='png', transforms=transform_x)
        val_ds = PolypSegm("datasets/TestDataset/ETIS-LaribPolypDB", annot_extension='png', im_extension='png',
                           transforms=transform_x_test)
        test_ds = PolypSegm("datasets/TestDataset/ETIS-LaribPolypDB", annot_extension='png', im_extension='png',
                            transforms=transform_x_test)
    else:
        raise NotImplemented("Dataset not valid")
    return num_classes, test_ds, train_ds, val_ds
