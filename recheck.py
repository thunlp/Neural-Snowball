import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader
import argparse
from torch.autograd import Variable
import numpy as np

def get_repre(model, data_loader, save_path):
    # repre = np.load('./_repre/cnn_encoder_on_fewrel.npy')
    repre = np.load('./_repre/cnn_encoder_on_fewrel.distant.npy')
    for it in range(data_loader.instance_tot):
        print(it)
        word = torch.from_numpy(data_loader.data_word[it:it+1]).long().cuda()
        pos1 = torch.from_numpy(data_loader.data_pos1[it:it+1]).long().cuda()
        pos2 = torch.from_numpy(data_loader.data_pos2[it:it+1]).long().cuda()
        batch = {'word': word, 'pos1': pos1, 'pos2': pos2} 
        batch_repre = model(batch).squeeze()
        batch_repre=(batch_repre.cpu().detach().numpy())
        if np.abs(batch_repre-repre[it]).sum() > 0.1:
            print('---')
            print(it)
            print(batch_repre)
            print(repre[data_loader.uid[it]])
            break

    print('')

max_length = 40
train_train_data_loader = DataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
train_val_data_loader = DataLoader('./data/train_val.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = DataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)
distant = DataLoader('./data/distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_train_data_loader.word_vec_mat, max_length)
ckpt_encoder = {}
for name, param in torch.load('./checkpoint/cnn_encoder_on_fewrel.pth.tar.bak')['state_dict'].items():
    if 'sentence_encoder' in name:
        ckpt_encoder[name[17:]] = param
encoder.load_state_dict(ckpt_encoder)
encoder.cuda()
encoder.eval()

get_repre(encoder, distant, './_repre/' + 'cnn_encoder_on_fewrel.' + 'distant' + '.npy')

# siamese = nrekit.sentence_encoder.CNNSentenceEncoder(train_train_data_loader.word_vec_mat, max_length)
# ckpt_siamese = {}
# for name, param in torch.load('./checkpoint/cnn_siamese_on_fewrel.pth.tar.bak')['state_dict'].items():
#     if 'sentence_encoder' in name:
#         ckpt_siamese[name[17:]] = param
# siamese.load_state_dict(ckpt_siamese)
# siamese.cuda()
# siamese.eval()
# 
# for data_loader, name in [(train_train_data_loader, 'train_train'), (train_val_data_loader, 'train_val'), (val_data_loader, 'val'), (test_data_loader, 'test'), (distant, 'distant')]:
#     get_repre(siamese, data_loader, './_repre/' + 'cnn_siamese_on_fewrel.' + name + '.npy')
# 
# 
