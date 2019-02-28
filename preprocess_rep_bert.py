import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader
import argparse
from torch.autograd import Variable
import numpy as np

def get_repre(model, data_loader, save_path):
    print('repre save to ' + save_path)
    repre = []
    batch_size = 32
    total_step = data_loader.instance_tot // batch_size
    if data_loader.instance_tot % batch_size != 0:
        total_step += 1
    for it in range(total_step):
        a = it * batch_size
        b = min((it + 1) * batch_size, data_loader.instance_tot)
        word = torch.from_numpy(data_loader.data_word[a:b]).long().cuda()
        mask = torch.from_numpy(data_loader.data_mask[a:b]).long().cuda()
        batch = {'word': word, 'mask': mask}
        batch_repre = model(batch)
        repre.append(batch_repre.cpu().detach().numpy())
        sys.stdout.write('[{0:3.2f}%] {1:6} / {2:6}'.format(100 * float(it) / float(total_step), it, total_step) + '\r')
        sys.stdout.flush()

    print('')
    repre = np.concatenate(repre, 0)
    np.save(save_path, repre)
    
max_length = 90
train_train_data_loader = DataLoader('./data/train_train.json', vocab='./data/bert_vocab.txt', max_length=max_length)
train_val_data_loader = DataLoader('./data/train_val.json', vocab='./data/bert_vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', vocab='./data/bert_vocab.txt', max_length=max_length)
test_data_loader = DataLoader('./data/test.json', vocab='./data/bert_vocab.txt', max_length=max_length)
distant = DataLoader('./data/distant.json', vocab='./data/bert_vocab.txt', max_length=max_length, distant=True)

encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
ckpt_encoder = {}
for name, param in torch.load('./checkpoint/bert_encoder_on_fewrel.pth.tar.bak')['state_dict'].items():
    if 'sentence_encoder' in name:
        ckpt_encoder[name[17:]] = param
encoder.load_state_dict(ckpt_encoder)
encoder.cuda()
encoder.eval()

for data_loader, name in [(train_train_data_loader, 'train_train'), (train_val_data_loader, 'train_val'), (val_data_loader, 'val'), (test_data_loader, 'test'), (distant, 'distant')]:
    get_repre(encoder, data_loader, './_repre_split/' + 'bert_encoder_on_fewrel.' + name + '.npy')

siamese = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
ckpt_siamese = {}
for name, param in torch.load('./checkpoint/bert_siamese_on_fewrel.pth.tar.bak')['state_dict'].items():
    if 'sentence_encoder' in name:
        ckpt_siamese[name[17:]] = param
siamese.load_state_dict(ckpt_siamese)
siamese.cuda()
siamese.eval()

for data_loader, name in [(train_train_data_loader, 'train_train'), (train_val_data_loader, 'train_val'), (val_data_loader, 'val'), (test_data_loader, 'test'), (distant, 'distant')]:
    get_repre(siamese, data_loader, './_repre_split/' + 'bert_siamese_on_fewrel.' + name + '.npy')


