import torch
import os
from datasets import DatasetDict
from torch.utils.data import DataLoader
from deep_fusion_model.model.FusionHead import FusionHead
def concat_cls(cls_bert,cls_vit,save_path):

    cls_concat = torch.cat([cls_vit, cls_bert], dim=-1)

    return cls_concat
def savemodel(model, best_state, best_thr, save_path="models_no_finetune/fusion_head.pt"):
    # créer le dossier parent s'il n'existe pas
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # on sauvegarde le BEST state, pas l'état courant
    state_to_save = best_state if best_state is not None else model.state_dict()

    torch.save({
        "model_state_dict": state_to_save,
        "best_thr": best_thr,
        "d_in_v": 768,
        "d_in_t": 768,
        "d": 512,
        "num_labels": 23,
        "dropout": 0.3
    }, save_path)

    print(f"✅ Modèle sauvegardé dans : {save_path}")


def load_fusion_head(ckpt_path="models_no_finetune/fusion_head.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(ckpt_path, map_location=device)

    # recréer le head avec les mêmes hyperparams que dans le checkpoint
    model = FusionHead(
        d_in_v=ckpt.get("d_in_v", 768),
        d_in_t=ckpt.get("d_in_t", 768),
        d=ckpt.get("d", 512),
        num_labels=ckpt.get("num_labels", 23),
        p=ckpt.get("dropout", 0.3)
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    best_thr = ckpt.get("best_thr", 0.3)
    return model, best_thr, device


def multilabel_metrics_from_logits(logits, targets, thr=0.5, eps=1e-9):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    tp = (preds*targets).sum().item()
    fp = (preds*(1-targets)).sum().item()
    fn = ((1-preds)*targets).sum().item()
    precision = tp/(tp+fp+eps); recall = tp/(tp+fn+eps)
    f1_micro = 2*precision*recall/(precision+recall+eps)
    return precision, recall, f1_micro

def dataset_spliter(ds):
    # Mélange
    # ds = ds.shuffle(seed=42)

    tmp = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = tmp["train"].shuffle(seed=42)  # 80%
    val_test = tmp["test"].shuffle(seed=42)  # 20%

    # on coupe les 20% en deux
    tmp2 = val_test.train_test_split(test_size=0.5, seed=42)
    val_ds = tmp2["train"].shuffle(seed=42)  # 10%
    test_ds = tmp2["test"].shuffle(seed=42) # 10%

    return DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds
    })

#Test git
def dataset_loader(ds_splits):

    train_ds = ds_splits["train"]
    val_ds = ds_splits["val"]
    test_ds = ds_splits["test"]

    # ⚙️ format torch
    train_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])
    val_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])
    test_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])

    # 📦 DataLoaders
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_loader, val_loader, test_loader


