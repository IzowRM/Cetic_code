import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset, load_from_disk
from safetensors.torch import load_model
from transformers import CLIPModel
from transformers import AutoProcessor
from model.MultiModalDataset import MultiModalDataset
from model.CLIPMultiLabelHF import CLIPMultiLabelHF
def clip_pipe_classification():
    model_id = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
    processor = AutoProcessor.from_pretrained(model_id)
    label_cols = [
        "drama", "comedy", "romance", "thriller", "crime", "action", "adventure", "horror",
        "documentary", "mystery", "sci-fi", "fantasy", "family", "biography", "war", "history",
        "music", "animation", "musical", "western", "sport", "short", "film-noir"
    ]
    num_labels = 23

    ds = load_from_disk("../data/ds_flat")

    dataset = ds.train_test_split(test_size=0.2, seed=42)

    ds_train = dataset["train"].shuffle(seed=42)
    val_df = dataset["test"].shuffle(seed=42)

    train_dataset = MultiModalDataset(
        ds_train,  # HF Dataset
        processor,
        text_col="plot",  # comme dans ton print
        img_col="image",  # comme dans ton print
    )

    val_dataset = MultiModalDataset(
        val_df,
        processor,
        text_col="plot",
        img_col="image",
    )
    from transformers import TrainingArguments, Trainer

    model = CLIPMultiLabelHF(model_id, num_labels=num_labels, freeze_clip=True)

    lr = 1e-4
    wd = 0.01

    training_args = TrainingArguments(
        output_dir="./clip_multilabel",
        per_device_train_batch_size=64,
        per_device_eval_batch_size=64,
        num_train_epochs=5,
        learning_rate=lr,
        weight_decay=wd,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
        remove_unused_columns=False,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        from sklearn.metrics import f1_score

        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)

        return {
            "f1_micro": f1_score(labels, preds, average="micro"),
            "f1_macro": f1_score(labels, preds, average="macro"),
        }

    # D'abord, tout geler (déjà fait dans __init__)
    for p in model.clip.parameters():
        p.requires_grad = False

    # Nombre de couches
    n_vision = len(model.clip.vision_model.encoder.layers)
    n_text = len(model.clip.text_model.encoder.layers)

    # Dégeler les 2 derniers blocs de l'encodeur image
    for i in range(n_vision - 2, n_vision):
        for p in model.clip.vision_model.encoder.layers[i].parameters():
            p.requires_grad = True

    # Dégeler les 2 derniers blocs de l'encodeur texte
    for i in range(n_text - 2, n_text):
        for p in model.clip.text_model.encoder.layers[i].parameters():
            p.requires_grad = True

    # (Optionnel) dégeler les projections finales
    for p in model.clip.visual_projection.parameters():
        p.requires_grad = True
    for p in model.clip.text_projection.parameters():
        p.requires_grad = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()   # par défaut sur eval_dataset
    print(metrics)

def main():
    clip_pipe_classification()


if __name__ == '__main__':
    main()
