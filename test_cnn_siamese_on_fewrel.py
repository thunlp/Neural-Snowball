import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader

from pytorch_pretrained_bert import BertAdam

max_length = 40
train_data_loader = DataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model = models.snowball_euc.Siamese(sentence_encoder, hidden_size=230)

model_name = 'cnn_siamese_euc_on_fewrel'

checkpoint = framework.__load_model__('./checkpoint/' + model_name + '.pth.tar.bak')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

framework.eval_siamese(model, eval_iter=500, threshold=0.2, s_num_size=4, s_num_class=8)
print('')
framework.eval_siamese(model, eval_iter=500, threshold=0.3, s_num_size=4, s_num_class=8)
print('')
framework.eval_siamese(model, eval_iter=500, threshold=0.4, s_num_size=4, s_num_class=8)
print('')
framework.eval_siamese(model, eval_iter=500, threshold=0.9, s_num_size=4, s_num_class=8)
print('')
