import torch
from datasets import DatasetDict
from torch.utils.data import DataLoader


def concat_cls(cls_bert,cls_vit):

    cls_concat = torch.cat([cls_vit, cls_bert], dim=-1)

    return cls_concat

def multilabel_metrics_from_logits(logits, targets, thr=0.5, eps=1e-9):
    probs = torch.sigmoid(logits)
    preds = (probs >= thr).float()
    tp = (preds*targets).sum().item()
    fp = (preds*(1-targets)).sum().item()
    fn = ((1-preds)*targets).sum().item()
    precision = tp/(tp+fp+eps); recall = tp/(tp+fn+eps)
    f1 = 2*precision*recall/(precision+recall+eps)
    return precision, recall, f1

def dataset_spliter(ds):
    # Mélange
    ds = ds.shuffle(seed=42)

    # 10% test
    tmp = ds.train_test_split(test_size=0.10, seed=42)
    train_val = tmp["train"]
    test_ds = tmp["test"]

    # 10% val sur le reste (0.1 / 0.9 = 0.1111)
    tmp2 = train_val.train_test_split(test_size=0.1111, seed=42)
    train_ds = tmp2["train"]
    val_ds = tmp2["test"]

    return DatasetDict({
        "train": train_ds,
        "val": val_ds,
        "test": test_ds
    })


def dataset_loader(ds_splits):

    train_ds = ds_splits["train"]
    val_ds = ds_splits["val"]
    test_ds = ds_splits["test"]

    # ⚙️ format torch
    train_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])
    val_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])
    test_ds.set_format(type="torch", columns=["cls_vit", "cls_bert", "labels"])

    # 📦 DataLoaders
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    return train_loader, val_loader, test_loader


