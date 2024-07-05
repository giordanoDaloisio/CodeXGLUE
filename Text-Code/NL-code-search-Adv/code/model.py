# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from icecream import ic


    
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
            return code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,code_vec,nl_vec

      
        
 
