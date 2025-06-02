from transformers import AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("saved_models_t5")

torch.save(model.state_dict(), "saved_models_t5/model.bin")