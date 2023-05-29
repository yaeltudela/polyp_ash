from collections import defaultdict

import torch

from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler


def reduce_param_groups(params):
    # Reorganize the parameter groups and merge duplicated groups.
    # The number of parameter groups needs to be as small as possible in order
    # to efficiently use the PyTorch multi-tensor optimizer. Therefore instead
    # of using a parameter_group per single parameter, we reorganize the
    # parameter groups and merge duplicated groups. This approach speeds
    # up multi-tensor optimizer significantly.

    ret = defaultdict(dict)
    for item in params:
        assert "params" in item
        cur_params = {x: y for x, y in item.items() if x != "params"}
        for param in item["params"]:
            ret[param].update({"params": [param], **cur_params})

    params = list(ret.values())
    groups = defaultdict(list)  # re-group all parameter groups by their hyperparams
    for item in params:
        cur_params = tuple((x, y) for x, y in item.items() if x != "params")
        groups[cur_params].extend(item["params"])
    ret = []
    for param_keys, param_values in groups.items():
        cur = {kv[0]: kv[1] for kv in param_keys}
        cur["params"] = param_values
        ret.append(cur)
    return ret


def modify_params(model, weight_decay_skip_list=(), weight_decay_skip_keywords=(), lr_parameters=(), mod_lr=1):
    param_list = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias"):
            if check_keywords_in_name(name, lr_parameters):
                param_list.append({'params': [param], 'weight_decay': 0., 'lr': mod_lr})
            else:
                param_list.append({'params': [param], 'weight_decay': 0.})
        elif (name in weight_decay_skip_list) or check_keywords_in_name(name, weight_decay_skip_keywords):
            if check_keywords_in_name(name, lr_parameters):
                param_list.append({'params': [param], 'weight_decay': 0., 'lr': mod_lr})
            else:
                param_list.append({'params': [param], 'weight_decay': 0.})
        else:
            if check_keywords_in_name(name, lr_parameters):
                param_list.append({'params': [param], 'lr': mod_lr})
            else:
                param_list.append({'params': [param]})

    parameters = reduce_param_groups(param_list)
    return parameters


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin


def build_optimizer(model, lr, sgd=False):
    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    lr_mod = lr * 1e-2
    lr_keywords = ()
    parameters = modify_params(model, skip, skip_keywords, lr_keywords, lr_mod)
    if sgd:
        optimizer = torch.optim.SGD(parameters, lr=lr, weight_decay=0.01, momentum=0.9)
    else:
        optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, amsgrad=False)
        # optimizer = torch.optim.AdamW(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=1., amsgrad=False)

    return optimizer


def build_scheduler(optimizer, name, num_steps, lr):
    warmup_steps = int(num_steps * 0.05)
    warmup_lr = lr / 10

    lr_scheduler = None
    if name == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            cycle_mul=1.,
            lr_min=5e-6,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )

    elif name == 'linear':
        lr_scheduler = LinearLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min_rate=0.01,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )
    elif name == 'step':
        decay_steps = int(num_steps * 0.2)

        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_steps,
            decay_rate=0.1,
            warmup_lr_init=warmup_lr,
            warmup_t=warmup_steps,
            t_in_epochs=False,
        )

    return lr_scheduler


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return
