from datasets import load_from_disk
from sklearn.metrics import f1_score
from transformers import TrainingArguments, Trainer
import numpy as np
from fonctions.vit_tools import model_vit_for_finetune, pixel_value_method
from PIL import Image

def finetune_vit():
    #Charger les données
    model, processor = model_vit_for_finetune()
    ds = load_from_disk("../data/ds_flat")

    print(ds[34]['image'])
    ex0 = ds[34]["image"]
    print(type(ex0))

    # def pixel_value_method(exemple):
    #     image = exemple['image'][0]
    #
    #     enc = processor(images=image, return_tensors="pt")
    #     pixel_values = enc["pixel_values"]
    #
    #     exemple['pixel_values'] = pixel_values.squeeze(0).numpy()
    #
    #     return exemple

    def pixel_value_method(example):
        img = example["image"]  # JpegImageFile
        img = img.convert("RGB")  # au cas où

        enc = processor(img, return_tensors="pt")
        example["pixel_values"] = enc["pixel_values"][0].numpy()
        return example

    ds = ds.map(pixel_value_method)
    dataset = ds.train_test_split(test_size=0.2, seed=42)
    ds.set_format(type="torch", columns=["pixel_values", "labels"])
    train_dataset = dataset["train"].shuffle(seed=42)
    val_dataset = dataset["test"].shuffle(seed=42)
    #charger le model

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # logits: [batch, num_labels], labels: [batch, num_labels]
        probs = 1 / (1 + np.exp(-logits))  # sigmoid
        preds = (probs > 0.5).astype(int)  # threshold 0.5

        # F1 micro/macro multi-label
        f1_micro = f1_score(labels, preds, average="micro", zero_division=0)
        f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
        return {
            "f1_micro": f1_micro,
            "f1_macro": f1_macro,
        }

    training_args = TrainingArguments(
        output_dir="./vit-multilabel",
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        learning_rate=5e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="none",
        fp16=True,

    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    #save le model
    model.eval()







def main():
    finetune_vit()
if __name__ == '__main__':
    main()