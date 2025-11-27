from accelerate.test_utils.scripts.external_deps.test_pippy import test_bert
from datasets import load_from_disk
from torch import nn
from torch.nn import Linear
import torch.nn.functional as F

from fonctions.FocalLossMultiLabel import FocalLossMultiLabel
from fonctions.bert_tools import model_bert, bert_features, bert_loader, bert_finetune_features
from fonctions.tools_for_main import concat_cls, dataset_spliter, multilabel_metrics_from_logits, \
    dataset_loader, savemodel
from fonctions.vit_tools import model_vit, viT_cls
# from ViT_CLS import viT_cls
import torch
from torch.optim import AdamW

from model.FusionHead import FusionHead


def main_bert_vit_no_finetune(ds,bert_model,vit_model):

    def mapped_fonction(exemple):

        text_list = exemple['input_ids']
        image = exemple['image'].convert("RGB")
        label = exemple['labels']
        cls_vit = viT_cls(image, vit_model).squeeze(0).detach().cpu().numpy().astype("float32")
        cls_bert = bert_finetune_features(text_list, bert_model).squeeze(0).detach().cpu().numpy().astype("float32")

        return {
            'cls_vit': cls_vit,
            'cls_bert':cls_bert,
            'labels': label
        }

    return mapped_fonction

def data_set_creation():

    print('chargement des models_no_finetune')
    #import du  model:
    #Bert sans finetune
    # bert_model = model_bert()
    # Bert avec finetune
    bert_model = bert_loader()
    vit_model = model_vit().eval()

    print('Chargement du ds')
    ds = load_from_disk("../data/dataset_tokenized")
    mapped_fn = main_bert_vit_no_finetune(ds, bert_model, vit_model)
    ds2 = ds.map(mapped_fn)

    # 1) Garder seulement cls et labels
    cols_to_keep = ["cls_vit", "cls_bert", "labels"]
    cols_to_remove = [c for c in ds2.column_names if c not in cols_to_keep]

    ds_final = ds2.remove_columns(cols_to_remove)
    print(ds_final[0])
    #Creation du dataset:
    #Bert sans finetune
    # ds_final.save_to_disk("../data/dataset_cls_bert_vit")

    #Bert avec fintune:
    ds_final.save_to_disk("../data/dataset_cls_bert_finetune_vit_no_finetune")

def training_model():
    # ds = load_from_disk("../data/dataset_cls_bert_vit")
    ds = load_from_disk("../data/dataset_cls_bert_finetune_vit_no_finetune")

    train_loader, val_loader, test_loader = dataset_loader(dataset_spliter(ds))

    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))


    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FusionHead(d_in_v=768, d_in_t=768, d=512, num_labels=23, p=0.3).to(device)
    lr = 3e-5
    weight_decay = 2e-5
    thr = 0.3
    # Ajout de focal loss pour les réduire l'imact des classes domiante
    criterion = FocalLossMultiLabel(gamma=2.0, alpha=0.25)
    # criterion = nn.BCEWithLogitsLoss()  # multilabel
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # def train_one_epoch(model, loader):
    #     model.train()
    #     total_loss = 0.0
    #
    #     for batch in loader:
    #         cls_vit = batch["cls_vit"].to(device).float()
    #         cls_bert = batch["cls_bert"].to(device).float()
    #         labels = batch["labels"].to(device).float()
    #
    #         logits = model(cls_vit, cls_bert)  # [B,23]
    #         loss = criterion(logits, labels)
    #
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #
    #         total_loss += loss.item()
    #
    #     return total_loss / len(loader)
    tau = 0.07
    lam = 0.6

    def train_one_epoch(model, loader):
        model.train()
        total_loss = 0.0

        for batch in loader:
            cls_vit = batch["cls_vit"].to(device).float()
            cls_bert = batch["cls_bert"].to(device).float()
            labels = batch["labels"].to(device).float()

            # logits + projections (dans même espace)
            logits, z_v, z_t = model(cls_vit, cls_bert, return_proj=True)

            # ----- loss classification multilabel -----
            loss_cls = criterion(logits, labels)

            # ----- loss alignement contrastif -----
            # normalisation pour cosine
            z_vn = F.normalize(z_v, dim=-1)  # (B, d)
            z_tn = F.normalize(z_t, dim=-1)  # (B, d)

            # similarité cross-batch
            sim = (z_tn @ z_vn.T) / tau  # (B, B)

            # la paire correcte est sur la diagonale
            align_labels = torch.arange(sim.size(0), device=sim.device)

            loss_align = (
                                 F.cross_entropy(sim, align_labels) +
                                 F.cross_entropy(sim.T, align_labels)
                         ) / 2

            # ----- loss totale -----
            loss = loss_cls + lam * loss_align

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(model, loader, thr=thr):
        model.eval()
        total_loss = 0.0
        tp = fp = fn = 0.0


        for batch in loader:


            # ## Bert Suelement
            # cls_bert = batch["cls_bert"].to(device).float()
            # cls_vit_zeros = torch.zeros_like(cls_bert)
            # logits = model(cls_vit_zeros, cls_bert)

            # Vit Seulement
            # cls_vit = batch["cls_vit"].to(device).float()
            # cls_bert_zero = torch.zeros_like(cls_vit)
            # logits = model(cls_vit, cls_bert_zero)

            ## Les deux
            cls_vit = batch["cls_vit"].to(device).float()
            cls_bert = batch["cls_bert"].to(device).float()
            logits = model(cls_vit, cls_bert)

            labels = batch["labels"].to(device).float()
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # métriques
            probs = torch.sigmoid(logits)
            preds = (probs >= thr).float()
            tp += (preds * labels).sum().item()
            fp += (preds * (1 - labels)).sum().item()
            fn += ((1 - preds) * labels).sum().item()

        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)

        return total_loss / len(loader), precision, recall, f1

    best_f1 = 0.0
    best_state = None

    for epoch in range(1, 20):
        train_loss = train_one_epoch(model, train_loader)
        val_loss, p, r, f1 = evaluate(model, val_loader, thr=thr)

        print(
            f"Epoch {epoch} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | Precision={p:.3f} Recall={r:.3f} F1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    test_loss, p, r, f1 = evaluate(model, test_loader, thr=thr)
    print(f"TEST | loss={test_loss:.4f} | P={p:.3f} R={r:.3f} F1={f1:.3f}")

    best_thr, best_f1 = 0.5, 0
    for thr in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        _, p, r, f1 = evaluate(model, val_loader, thr=thr)
        print(thr, p, r, f1)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print("Best thr:", best_thr, "Best val F1:", best_f1)
    # savemodel(model, best_state, best_thr)

if __name__ == "__main__":
    # data_set_creation()

    training_model()

