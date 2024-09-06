from torchinfo import summary
from models import Model, distill_loss
from transformers import AdamW, get_linear_schedule_with_warmup, RobertaConfig, RobertaModel

config = RobertaConfig.from_pretrained("microsoft/codebert-base")
config.num_attention_heads = 8
config.hidden_size = 96
config.intermediate_size = 64
config.vocab_size = 1000
config.num_hidden_layers = 12
config.hidden_dropout_prob = 0.5
model = Model(RobertaModel(config=config))

print(model)