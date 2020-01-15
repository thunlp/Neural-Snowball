import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader
import argparse

max_length = 90
train_data_loader = DataLoader('./data/train.json', vocab='./data/bert_vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', vocab='./data/bert_vocab.txt', max_length=max_length)
test_data_loader = DataLoader('./data/test.json', vocab='./data/bert_vocab.txt', max_length=max_length)
distant = DataLoader('./data/distant.json', vocab='./data/bert_vocab.txt', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_data_loader, val_data_loader, test_data_loader, distant)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
sentence_encoder2 = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')

model2 = models.snowball.Siamese(sentence_encoder2, hidden_size=768)
model = models.snowball.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=model2, hidden_size=768)

# load pretrain
checkpoint = torch.load('./checkpoint/bert_encoder_on_fewrel.pth.tar.bak')['state_dict']
checkpoint2 = torch.load('./checkpoint/bert_siamese_on_fewrel.pth.tar.bak')['state_dict']
for key in checkpoint2:
    checkpoint['siamese_model.' + key] = checkpoint2[key]
model.load_state_dict(checkpoint)
model.cuda()
model.train()
model_name = 'bert_snowball'

# eval
# framework.train(model, model_name, model2=model2)
# framework.eval_siamese(model2, threshold=0.99)
# print('')

# framework.eval_baseline(model, support_size=5, query_class=10)
framework.eval(model, support_size=10, query_class=10)
