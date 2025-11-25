import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


class CLIPMultiLabelHF(nn.Module):
    def __init__(self, model_name: str, num_labels: int, hidden_dim: int = 512, freeze_clip: bool = True):
        super().__init__()

        self.clip = CLIPModel.from_pretrained(model_name)
        emb_dim = self.clip.config.projection_dim  # dim des image_embeds / text_embeds
        self.num_labels = num_labels

        # Option : geler CLIP (linear probe / fine-tune léger)
        if freeze_clip:
            for p in self.clip.parameters():
                p.requires_grad = False

        # Tête multi-label
        self.classifier = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            labels=None,
    ):
        # Forward CLIP
        outputs = self.clip(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        img_embeds = outputs.image_embeds  # [B, D]
        txt_embeds = outputs.text_embeds  # [B, D]

        joint = torch.cat([img_embeds, txt_embeds], dim=-1)  # [B, 2D]
        logits = self.classifier(joint)  # [B, num_labels]

        loss = None
        if labels is not None:
            labels = labels.float()
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return {"loss": loss, "logits": logits}
