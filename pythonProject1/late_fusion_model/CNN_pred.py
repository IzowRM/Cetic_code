from datasets import load_from_disk, DatasetDict
from late_fusion_model.fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.Data_loader import data_cnn_treatment,hot_to_labels, data_cnn_spliter, hf_transform, collate_resize
from late_fusion_model.fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.inferance_settings import inf_cnn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader
import torch

def cnn_pred(ds_val):

    ds = data_cnn_treatment(ds_val)

    print('data_cnn_spliter')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_loader = DataLoader(ds,
                             batch_size=32,
                             shuffle=False,
                             num_workers=0,
                             collate_fn=collate_resize)

    model = cnn_loader(device)

    print('model load')

    # lire les entrées
    # données les résultat
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
            break  # juste pour tester sur un exemple