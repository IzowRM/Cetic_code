#faire les imports
from datasets import load_dataset
from mpmath import sigmoid
from wandb.integration.torch.wandb_torch import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from late_fusion_model.fonctions.model_loader import Bert_loader
from late_fusion_model.fonctions.inferance_settings import inf
from late_fusion_model.fonctions.Data_loader import flatten_text_only , extract_plot, data_loader
#Charger les deux model
    #model Bert

def main_Bert():
    print('Chargement du ds')
    ds = load_dataset("parquet", data_files="../data/mmimdb_merged.parquet", split="train")
    ds_flat = ds.map(flatten_text_only, remove_columns=ds.column_names)
    ds_flat = ds_flat.map(lambda batch: {"plot": [extract_plot(t) for t in batch["prompt"]]},
                          batched=True)

    ds_flat = data_loader(ds_flat)
    print(ds_flat[0])
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print('Creation du Tokenizer')
    def tokenize_function(examples):
        return tokenizer(examples["plot"],
                         padding="max_length",
                         truncation=True)
    #

    print('step1')

    tokenized_datasets = ds_flat.map(tokenize_function, batched=True)

    dataset = tokenized_datasets.train_test_split(test_size=0.2, seed=42)
    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42)
    print(eval_dataset[0])

    model = Bert_loader()
    model.eval()

    print(model.classifier.weight.shape)
    inf(eval_dataset,model,tokenizer)
    print('Finis')
if __name__ == "__main__":
    main_Bert()
#merge les proba


