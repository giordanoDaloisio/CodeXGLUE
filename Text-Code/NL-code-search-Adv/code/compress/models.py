"""Only Model, biLSTM and distill_loss are used in experiments."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss


class Model(nn.Module):   
    def __init__(self, encoder):
        super(Model, self).__init__()
        self.encoder = encoder
        #self.config=config
        #self.tokenizer=tokenizer
        #self.args=args
    
        
    def forward(self, code_inputs,nl_inputs,return_vec=False): 
        bs=code_inputs.shape[0]
        inputs=torch.cat((code_inputs,nl_inputs),0)
        outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        code_vec=outputs[:bs]
        nl_vec=outputs[bs:]
        # if self.args.model_type == 'roberta':
        #     outputs=self.encoder(inputs,attention_mask=inputs.ne(1))[1]
        #     code_vec=outputs[:bs]
        #     nl_vec=outputs[bs:]
        # else:
        #     outputs=self.encoder(inputs,attention_mask=inputs.ne(1))
        #     hidden_states = outputs.logits
        #     code_vec = hidden_states[:bs, 0, :]
        #     nl_vec = hidden_states[bs:, 0, :]
        
        if return_vec:
            return outputs,code_vec,nl_vec
        scores=(nl_vec[:,None,:]*code_vec[None,:,:]).sum(-1)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
        return loss,outputs,code_vec,nl_vec



def distill_loss(logits, knowledge, temperature=10.0):

    loss = F.kl_div(F.log_softmax(logits/temperature), F.softmax(knowledge /
                    temperature), reduction="batchmean") * (temperature**2)
    # Equivalent to cross_entropy for soft labels, from https://github.com/huggingface/transformers/blob/50792dbdcccd64f61483ec535ff23ee2e4f9e18d/examples/distillation/distiller.py#L330

    return loss

