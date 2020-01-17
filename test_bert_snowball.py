import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader
import argparse
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--shot', default=5, type=int, 
        help='Number of seeds')
parser.add_argument('--eval_iter', default=1000, type=int, 
        help='Eval iteration')
args = parser.parse_args()

max_length = 90
train_train_data_loader = DataLoader('./data/train_train.json', vocab='./data/bert-base-uncased/vocab.txt', max_length=max_length)
train_val_data_loader = DataLoader('./data/train_val.json', vocab='./data/bert-base-uncased/vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', vocab='./data/bert-base-uncased/vocab.txt', max_length=max_length)
test_data_loader = DataLoader('./data/val.json', vocab='./data/bert-base-uncased/vocab.txt', max_length=max_length)
distant = DataLoader('./data/distant.json', vocab='./data/bert-base-uncased/vocab.txt', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_val_data_loader, val_data_loader, test_data_loader, distant)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
sentence_encoder2 = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')

model2 = models.snowball.Siamese(sentence_encoder2, hidden_size=768)
model = models.snowball.Snowball(sentence_encoder, base_class=train_train_data_loader.rel_tot, siamese_model=model2, hidden_size=768, neg_loader=train_train_data_loader)

# load pretrain
checkpoint = torch.load('./checkpoint/bert_encoder_on_fewrel.pth.tar')['state_dict']
checkpoint2 = torch.load('./checkpoint/bert_siamese_on_fewrel.pth.tar')['state_dict']
for key in checkpoint2:
    checkpoint['siamese_model.' + key] = checkpoint2[key]
model.load_state_dict(checkpoint)
model.cuda()
model.train()
model_name = 'bert_snowball'

res = framework.eval(model, support_size=args.shot, query_size=50, eval_iter=args.eval_iter)
res_file = open('exp_bert_{}shot.txt'.format(args.shot), 'a')
res_file.write(res + '\n')
print('\n########## RESULT ##########')
print(res)
