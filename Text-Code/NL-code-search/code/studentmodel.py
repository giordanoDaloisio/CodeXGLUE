import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

class StudentModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(StudentModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

    def forward(self, code_inputs, nl_inputs, teacher_outputs=None): 
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]

        scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(scores, torch.arange(bs, device=scores.device))
     
        if teacher_outputs is not None:
            teacher_scores = F.softmax(teacher_outputs, dim=-1)
            student_scores = F.softmax(scores , dim=-1)
            # print(teacher_scores.shape, student_scores.shape)
            distillation_loss = F.kl_div(student_scores, teacher_scores, reduction='batchmean')
            loss += distillation_loss  
        
        return loss, code_vec, nl_vec