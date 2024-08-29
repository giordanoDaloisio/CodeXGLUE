# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import json
js_all=json.load(open('function.json'))

train_label_index=set()
train_unlabel_index=set()


with open('train.txt') as f:
    for i, line in enumerate(f):
        line=line.strip()
        if i%2 == 0:
            train_label_index.add(int(line))
        else:
            train_unlabel_index.add(int(line))
                    


        
        
with open('train_label.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_label_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')

with open('train_unlabel.jsonl','w') as f:
    for idx,js in enumerate(js_all):
        if idx in train_unlabel_index:
            js['idx']=idx
            f.write(json.dumps(js)+'\n')
            

