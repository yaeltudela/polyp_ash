import torch
from torch._six import inf


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, clip_grad=None, scale=True):
        self.clip_grad = clip_grad
        self.scale = scale
        if self.scale:
            self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, parameters=None, create_graph=False, update_grad=True):
        if self.scale:
            self._scaler.scale(loss).backward()
        else:
            loss.backward()
        if update_grad:
            if self.scale:
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place

            if self.clip_grad is not None:
                assert parameters is not None
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.clip_grad, norm_type=2.0,
                                                           error_if_nonfinite=True)
            else:
                grad_norm = ampscaler_get_grad_norm(parameters)
            if self.scale:
                self._scaler.step(optimizer)
                self._scaler.update()
            else:
                optimizer.step()
        else:
            grad_norm = None
        return grad_norm

    def state_dict(self):
        return self._scaler.state_dict() if self.scale else None

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
