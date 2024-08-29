"""Only Model, biLSTM and distill_loss are used in experiments."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args
    
        
    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        if self.args.model_type == 'roberta':
            outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
            code_vec=outputs[:bs]
            nl_vec=outputs[bs:]
        else:
            outputs=self.encoder(inputs,attention_mask=inputs.ne(1))
            hidden_states = outputs.logits  # Use the last hidden state
            # Assuming we take the [CLS] token representation as the sentence embedding
            # For DistilBERT, [CLS] token is the first token
            code_vec = hidden_states[:bs, 0, :]
            nl_vec = hidden_states[bs:, 0, :]
        
        if return_vec:
            return outputs,code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,outputs,code_vec,nl_vec




class biLSTM(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=hidden_dim,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.2)
        self.dense = nn.Linear(hidden_dim * 2, 200)
        self.fc = nn.Linear(200, n_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        outputs, (hidden, _) = self.lstm(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        x = F.relu(self.dense(hidden))
        x = self.dropout(x)
        logits = self.fc(x)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, prob
        else:
            return prob



class biGRU(nn.Module):
    def __init__(self, vocab_size, input_dim, hidden_dim, n_labels, n_layers):
        super(biGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, input_dim)
        self.gru = nn.GRU(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=n_layers,
                          batch_first=True,
                          bidirectional=True,
                          dropout=0.2)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, input_ids, labels=None):
        embed = self.embedding(input_ids)
        _, hidden = self.gru(embed)
        hidden = hidden.permute(1, 0, 2)
        hidden = torch.cat((hidden[:, -1, :], hidden[:, -2, :]), dim=1)
        logits = self.fc(hidden)
        prob = F.softmax(logits)

        if labels is not None:
            labels = labels.long()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, (1 - labels))
            return loss, prob
        else:
            return prob


def distill_loss(logits, knowledge, temperature=10.0):

    loss = F.kl_div(F.log_softmax(logits/temperature), F.softmax(knowledge /
                    temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss

