import models
import nrekit
import sys
import torch
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader


max_length = 40
train_data_loader = DataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/val_select.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = DataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)
distant = DataLoader('./data/distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_data_loader, val_data_loader, test_data_loader, distant)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
sentence_encoder2 = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)

model2 = models.snowball_siamese.Siamese(sentence_encoder2, hidden_size=230)
model = models.snowball_siamese.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=model2, hidden_size=230)

# load pretrain
checkpoint = torch.load('./checkpoint/cnn_encoder_on_fewrel.pth.tar.bak')['state_dict']
checkpoint2 = torch.load('./checkpoint/cnn_siamese_euc_on_fewrel.pth.tar.bak')['state_dict']
for key in checkpoint2:
    checkpoint['siamese_model.' + key] = checkpoint2[key]
model.load_state_dict(checkpoint)
model.cuda()
model.train()
model_name = 'cnn_snowball_euc'

# eval
# framework.train(model, model_name, model2=model2)
# framework.eval_siamese(model2, threshold=0.99)
# print('')
# framework.eval(model, support_size=5, query_class=16, query_size=600)
framework.eval_selected(model, threshold=0.99)
