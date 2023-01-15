import torch


class MaxNorm:
    def __init__(self, max_val=1.0) -> None:
        self.max_val = max_val

    def __call__(self, model):
        for name, param in model.named_parameters():
            if 'bias' in name or param.dim() < 2:
                continue
            with torch.no_grad():
                norm = param.norm(2, dim=1, keepdim=True).clamp(
                    min=self.max_val / 2)
                desired = torch.clamp(norm, max=self.max_val)
                param *= (desired / norm)
