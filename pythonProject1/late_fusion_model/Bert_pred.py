#faire les imports
from datasets import load_dataset
from mpmath import sigmoid
from wandb.integration.torch.wandb_torch import torch
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from late_fusion_model.fonctions.model_loader import Bert_loader
from late_fusion_model.fonctions.inferance_settings import inf
from late_fusion_model.fonctions.Data_loader import flatten_text_only , extract_plot, data_loader

def bert_pred(ds):
    #traitement des textes
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

    print('step1')
    tokenized_datasets = ds_flat.map(tokenize_function, batched=True)

    print(tokenized_datasets[0])

    model = Bert_loader()
    model.eval()

    print(model.classifier.weight.shape)
    inf(tokenized_datasets, model, tokenizer)
    print('Finis')