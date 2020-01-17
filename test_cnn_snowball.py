import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader
import argparse
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--shot', default=5, type=int, 
        help='Number of seeds')
parser.add_argument('--eval_iter', default=1000, type=int, 
        help='Eval iteration')

# snowball hyperparameter
parser.add_argument("--phase1_add_num", help="number of instances added in phase 1", type=int, default=5)
parser.add_argument("--phase2_add_num", help="number of instances added in phase 2", type=int, default=5)
parser.add_argument("--phase1_siamese_th", help="threshold of relation siamese network in phase 1", type=float, default=0.5)
parser.add_argument("--phase2_siamese_th", help="threshold of relation siamese network in phase 2", type=float, default=0.5)
parser.add_argument("--phase2_cl_th", help="threshold of relation classifier in phase 2", type=float, default=0.9)

parser.add_argument("--snowball_max_iter", help="number of iterations of snowball", type=int, default=5)

# fine-tune hyperparameter
parser.add_argument("--finetune_epoch", help="num of epochs when finetune", type=int, default=50)
parser.add_argument("--finetune_batch_size", help="batch size when finetune", type=int, default=10)
parser.add_argument("--finetune_lr", help="learning rate when finetune", type=float, default=0.05)
parser.add_argument("--finetune_wd", help="weight decay rate when finetune", type=float, default=1e-5)
parser.add_argument("--finetune_weight", help="loss weight of negative samples", type=float, default=0.2)

# inference batch_size
parser.add_argument("--infer_batch_size", help="batch size when inference", type=int, default=0)

# print
parser.add_argument("--print_debug", help="print debug information", action="store_true")
parser.add_argument("--eval", help="eval during snowball", action="store_true")

args = parser.parse_args()

max_length = 40
train_train_data_loader = DataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
train_val_data_loader = DataLoader('./data/train_val.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = DataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
distant = DataLoader('./data/distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_val_data_loader, val_data_loader, test_data_loader, distant)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_val_data_loader.word_vec_mat, max_length)
sentence_encoder2 = nrekit.sentence_encoder.CNNSentenceEncoder(train_val_data_loader.word_vec_mat, max_length)

model2 = models.snowball.Siamese(sentence_encoder2, hidden_size=230)
model = models.snowball.Snowball(sentence_encoder, base_class=train_train_data_loader.rel_tot, siamese_model=model2, hidden_size=230, neg_loader=train_train_data_loader, args=args)

# load pretrain
checkpoint = torch.load('./checkpoint/cnn_encoder_on_fewrel.pth.tar')['state_dict']
checkpoint2 = torch.load('./checkpoint/cnn_siamese_on_fewrel.pth.tar')['state_dict']
for key in checkpoint2:
    checkpoint['siamese_model.' + key] = checkpoint2[key]
model.load_state_dict(checkpoint)
model.cuda()
model.train()
model_name = 'cnn_snowball'

res = framework.eval(model, support_size=args.shot, query_size=50, eval_iter=args.eval_iter)
res_file = open('exp_cnn_{}shot.txt'.format(args.shot), 'a')
res_file.write(res + '\n')
print('\n########## RESULT ##########')
print(res)
