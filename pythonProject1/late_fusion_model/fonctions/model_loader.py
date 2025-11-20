from transformers import BertForSequenceClassification
import torch
from late_fusion_model.fonctions.NET import Net
def ask_int():
    while True:
        valeur = input("give an int : ")
        try:
            return int(valeur)
        except ValueError:
            print("that's not an int, try again.")

def Bert_loader():

    model = BertForSequenceClassification.from_pretrained("../data/BertModel/model")
    state = torch.load("../data/BertModel/classifier_head.pt", map_location="cpu")
    model.classifier.load_state_dict(state)
    model.config.problem_type = "multi_label_classification"

    return model

def cnn_loader(device):

    ckpt_path = '../data/CNN_model/models_image/best.pt'
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("Checkpoint chargé depuis l’epoch :", checkpoint["epoch"])
    num_labels = checkpoint["num_labels"]
    model = Net(num_labels=num_labels).to(device)

    # 4. Charger les poids
    model.load_state_dict(checkpoint["model_state"])

    # 5. Mode évaluation
    model.eval()

    return model
