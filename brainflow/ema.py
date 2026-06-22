import torch

class EMA:
    def __init__(self, model, decay=0.9995):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach().float()
            for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def apply(self, model):
        dtype = next(model.parameters()).dtype
        device = next(model.parameters()).device
        model.load_state_dict({
            k: v.to(dtype).to(device) for k, v in self.shadow.items()
        })

    def restore(self, sd, model):
        model.load_state_dict(sd)
