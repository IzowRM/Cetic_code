import torch.nn.functional as F
from deep_fusion_model.fonctions.vit_tools import pad_to_224_width
from datasets import load_from_disk
from PIL import ImageOps
import torch
from transformers import ViTModel, ViTImageProcessor
def viT_cls(image):

    img = image
    print('chargement du model ')
    model_name = "google/vit-base-patch16-224"
    processor = ViTImageProcessor.from_pretrained(model_name)
    vit = ViTModel.from_pretrained(model_name)
    vit.eval()  # pour éviter le dropout en inference


    img = img.convert("RGB")
    inputs = processor(images=img, return_tensors="pt")

    # 4) Passage dans ViT
    with torch.no_grad():
        outputs = vit(**inputs)   # ou vit(pixel_values=inputs["pixel_values"])

    # 5) Récupérer le CLS
    last_hidden_state = outputs.last_hidden_state          # (B, seq_len, hidden)
    cls_embedding = last_hidden_state[:, 0, :]             # (B, hidden)

    return cls_embedding

