import copy
import os.path
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F


def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--dataset', default='polyp_ash')
    ap.add_argument('--use_ass', action='store_true')
    ap.add_argument('--only_polyp', action='store_true')

    ap.add_argument('--im_size', default=224, type=int)
    ap.add_argument('--batch_size', default=2, type=int)
    ap.add_argument("--epochs", default=35, type=int)
    ap.add_argument("--accum_iter", default=1, type=int)
    ap.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])

    ap.add_argument("--config", default='tiny')
    ap.add_argument('--use_fpn', action='store_true')
    ap.add_argument('--decoder_name', type=str, default='swin_fpn')
    ap.add_argument('--head_name', type=str, default='base')

    ap.add_argument('--loss', type=str, default='cross_entropy')
    ap.add_argument('--extra_losses', nargs='+', type=str, choices=('iou', 'cls', 'cons'), default=(), )
    ap.add_argument("--lr", default=5e-6, type=float)
    ap.add_argument("--scheduler_type", default='', choices=['coseine', 'linear', 'step', ''])

    ap.add_argument("--output", default='', type=str)
    ap.add_argument("--resume", action='store_true')

    ap.add_argument('--label_smoothing', default=0.0, type=float)
    ap.add_argument('--use_mixup', action='store_true')

    ap.add_argument("--num_workers", default=4, type=int)
    ap.add_argument('--amp', action='store_true')

    parsed = ap.parse_args()

    return parsed


def seed_everything(seed=873):
    random.seed(seed)
    np.random.seed(seed)
    cpu_seed = torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)

    return cpu_seed


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=1e-4, patience=5, percentage=False, save=False):
        self.mode = mode
        self.min_delta = min_delta
        self.max_patience = patience
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.save = save
        self.state_dict = None

    def __call__(self, metrics, model):

        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            print("Error in metrics")
            return True

        # Update best_metric with no early stopping
        if self.patience is None:
            if self.is_better(metrics, self.best):
                self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.reset_bad_epochs()
            self.best = metrics

            if self.save and metrics <= self.best:
                self.state_dict = copy.deepcopy(model.state_dict())

        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def reset_bad_epochs(self):
        self.num_bad_epochs = 0

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                        best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                        best * min_delta / 100)

    def load_best_weights(self, model):
        if self.save and self.state_dict:
            model.load_state_dict(self.state_dict)
        return model

    @property
    def tol(self):
        if self.patience is None:
            return 1000
        return self.patience - self.num_bad_epochs


def setup():
    args = parse_args()
    print(args)
    torch_gen = seed_everything(873)
    return args, torch_gen


def is_pycharm():
    from os import getenv
    return getenv('PYCHARM_HOSTED') is not None


def is_debug():
    from sys import gettrace
    return gettrace() is not None


def resize(preds, size, mode='bilinear'):
    if preds.ndim < 4:
        preds = F.interpolate(preds.unsqueeze(1), size=size, mode=mode).squeeze(1)
    else:
        preds = F.interpolate(preds, size=size, mode=mode)

    return preds
