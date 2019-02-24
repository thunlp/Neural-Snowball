import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader
import argparse
from torch.autograd import Variable

def get_repre(model, data_loader, save_path):
    print('repre save to ' + save_path)
    repre = []
    batch_size = 10000 
    total_step = data_loader.instance_tot // batch_size
    if data_loader.instance_tot % batch_size != 0:
        total_step += 1
    for it in range(total_step):
        a = it * batch_size
        b = min((it + 1) * batch_size, data_loader.instance_tot)
        word = torch.from_numpy(data_loader.data_word[a:b]).cuda()
        pos1 = torch.from_numpy(data_loader.data_pos1[a:b]).cuda()
        pos2 = torch.from_numpy(data_loader.data_pos2[a:b]).cuda()
        batch = {'word': word, 'pos1': pos1, 'pos2': pos2} 
        batch_repre = model(batch)
        repre.append(batch_repre)
        sys.stdout.write('[{0:3.2f}%] {1:6} / {2:6}'.format(100 * float(it) / float(total_step), it, total_step) + '\r')
        sys.stdout.flush()

    print('')
    repre = np.concatenate(repre, 0)
    np.save(save_path, repre)

max_length = 40
train_train_data_loader = DataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
train_val_data_loader = DataLoader('./data/train_val.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = DataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)
distant = DataLoader('./data/distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_train_data_loader.word_vec_mat, max_length)
ckpt_encoder = torch.load('./checkpoint/cnn_encoder_on_fewrel.pth.tar.bak')['state_dict']
encoder.load_state_dict(ckpt_encoder)
encoder.cuda()
encoder.eval()

for data_loader, name in [(train_train_data_loader, 'train_train'), (train_val_data_loader, 'train_val'), (val_data_loader, 'val'), (test_data_loader, 'test')]:
    get_repre(encoder, data_loader, './_repre/' + 'cnn_encoder_on_fewrel.' + name + '.npy')

siamese = nrekit.sentence_encoder.CNNSentenceEncoder(train_train_data_loader.word_vec_mat, max_length)
ckpt_siamese = torch.load('./checkpoint/cnn_siamese_euc_on_fewrel.pth.tar.bak')['state_dict']
siamese.load_state_dict(ckpt_siamese)
siamese.cuda()
siamese.eval()

for data_loader, name in [(train_train_data_loader, 'train_train'), (train_val_data_loader, 'train_val'), (val_data_loader, 'val'), (test_data_loader, 'test')]:
    get_repre(siamese, data_loader, './_repre/' + 'cnn_siamese_on_fewrel.' + name + '.npy')


