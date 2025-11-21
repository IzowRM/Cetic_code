import torch
from torch import nn


class FusionHead(nn.Module):
    def __init__(self, d_in_v=768, d_in_t=768, d=512, num_labels=23, p=0.2):
        super().__init__()
        self.proj_v = nn.Linear(d_in_v, d)
        self.proj_t = nn.Linear(d_in_t, d)
        self.ln_v = nn.LayerNorm(d)
        self.ln_t = nn.LayerNorm(d)
        self.drop = nn.Dropout(p)

        self.gate = nn.Linear(2*d, d)  # pour g
        self.head = nn.Linear(d, num_labels)

    def forward(self, cls_vit, cls_bert):
        z_v = self.drop(self.ln_v(self.proj_v(cls_vit)))
        z_t = self.drop(self.ln_t(self.proj_t(cls_bert)))

        g = torch.sigmoid(self.gate(torch.cat([z_v, z_t], dim=-1)))
        z = z_t + g * z_v  # texte principal, image en complément

        return self.head(z)
