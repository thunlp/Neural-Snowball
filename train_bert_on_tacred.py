import models
import nrekit
import sys
import numpy as np
import torch
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader

from pytorch_pretrained_bert import BertAdam

max_length = 300
train_data_loader = DataLoader('./data/tacred_train.json', vocab='./data/bert_vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/tacred_val.json', vocab='./data/bert_vocab.txt', max_length=max_length, rel2id=train_data_loader.rel2id, shuffle=False)

weight_table = np.zeros((train_data_loader.rel_tot), dtype=np.float32)
for i in range(train_data_loader.rel_tot):
    scope = train_data_loader.rel2scope[train_data_loader.id2rel[i]]
    weight_table[i] = 1 / ((scope[1] - scope[0]) ** 0.05)
weight_table[13] *= 0.1
weight_table = torch.FloatTensor(weight_table).cuda()

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
model = models.snowball.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=None, hidden_size=768, drop_rate=0.1, weight_table=weight_table)
model.NA_label = 13
model_name = 'bert_encoder_on_tacred'

# pre-train
checkpoint = torch.load('./checkpoint/bert_encoder_on_fewrel.pth.tar.bak')['state_dict']
own_state = model.state_dict()
for name, param in checkpoint.items():
    if 'fc' not in name:
        own_state[name].copy_(param)

# set optimizer
batch_size = 4
train_epoch = 10

param_optimizer = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = BertAdam(optimizer_grouped_parameters, lr=5e-5)

framework.train_encoder_epoch(model, model_name, optimizer=optimizer, batch_size=batch_size, train_epoch=train_epoch, learning_rate=2e-5, warmup=True, grad_iter=8)
