from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd  # seulement si tu utilises aussi des DataFrame

class MultiModalDataset(Dataset):
    def __init__(
        self,
        data,
        processor,
        text_col="plot",      # <- important
        img_col="image",      # <- important
    ):
        self.data = data              # peut être un HF Dataset OU un DataFrame
        self.processor = processor
        self.text_col = text_col
        self.img_col = img_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # HF Dataset (dict) ou DataFrame (Series)
        if isinstance(self.data, pd.DataFrame):
            row = self.data.iloc[idx]
        else:
            row = self.data[idx]   # HuggingFace Dataset → dict

        # --- TEXTE ---
        text = row[self.text_col]      # ici "plot"

        # --- IMAGE ---
        img_data = row[self.img_col]   # ici "image" (PIL déjà)
        if isinstance(img_data, str):
            # au cas où tu as un chemin plus tard
            image = Image.open(img_data).convert("RGB")
        else:
            image = img_data           # déjà un PIL.Image.Image

        # --- LABELS ---
        labels = torch.tensor(row["labels"], dtype=torch.float32)

        # Processor CLIP
        encoded = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # enlever B=1
        encoded = {k: v.squeeze(0) for k, v in encoded.items()}
        encoded["labels"] = labels

        return encoded

