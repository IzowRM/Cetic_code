from datasets import load_from_disk, DatasetDict
from late_fusion_model.fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.Data_loader import data_cnn_treatment,hot_to_labels, data_cnn_spliter, hf_transform, collate_resize
from late_fusion_model.fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.inferance_settings import inf_cnn

# CNN_pred.py
from torch.utils.data import DataLoader
import torch
from late_fusion_model.fonctions.Data_loader import (
    data_cnn_treatment, hot_to_labels, hf_transform, collate_resize
)

def cnn_pred(ds_val):
    # 1) Appliquer le remap des labels (ça conserve les autres colonnes)
    ds = data_cnn_treatment(ds_val)  # contient encore les images

    # 2) Très important : produire 'pixel_values' à la volée
    ds = ds.with_transform(hf_transform)  # <-- AJOUT

    print('data_cnn_spliter')  # (ce print peut rester même si tu ne splittes pas ici)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(
        ds, batch_size=32, shuffle=False, num_workers=0, collate_fn=collate_resize
    )

    model = cnn_loader(device)
    print('model load')

    with torch.no_grad():
        for batch in test_loader:
            images = batch["pixel_values"][1].unsqueeze(0).to(device)
            logits = model(images)
            active, probs = inf_cnn(logits, thr=0.5)
            print(batch.keys())
            print(batch["pixel_values"].shape)
            if "labels" in batch:
                print(batch["labels"][1])

            hot_to_labels(batch["labels"][1])
            print("Labels prédits :", active)
            print("Probas :", probs)
            break
