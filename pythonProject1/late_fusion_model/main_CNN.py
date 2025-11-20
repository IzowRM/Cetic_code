from datasets import load_from_disk, DatasetDict
from fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.Data_loader import data_cnn_treatment,hot_to_labels, data_cnn_spliter, hf_transform, collate_resize
from late_fusion_model.fonctions.model_loader import cnn_loader
from late_fusion_model.fonctions.inferance_settings import inf_cnn
from torchvision import transforms
from PIL import Image
from torch.utils.data import DataLoader

import torch
def main_CNN():
    ds = load_from_disk("../data/ds_flat_labeled")
    print('data_cnn_treatment')

    ds = data_cnn_treatment(ds)

    print('data_cnn_spliter')

    ds = data_cnn_spliter(ds)
    print('DatasetDict')


    ds  = DatasetDict({
    "train": ds["train"].with_transform(hf_transform),
    "val":   ds["val"].with_transform(hf_transform),
    "test":  ds["test"].with_transform(hf_transform),})

    print(ds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(ds["train"], batch_size=4, shuffle=True,
                              num_workers=0, collate_fn=collate_resize)
    val_loader = DataLoader(ds["val"], batch_size=4, shuffle=False,
                            num_workers=0, collate_fn=collate_resize)
    test_loader = DataLoader(ds["test"], batch_size=4, shuffle=False,
                             num_workers=0, collate_fn=collate_resize)

    model = cnn_loader(device)
    print('model load')

    #lire les entrées
    #données les résultat
    with torch.no_grad():
        for batch in test_loader:

            # on prend le premier élément du batch, comme dans ton entraînement
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

if __name__ == "__main__":
    main_CNN()

