import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)

framework = nrekit.framework.Framework(train_data_loader, val_data_loader, test_data_loader)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model = models.finetune.Finetune(sentence_encoder, base_class=train_data_loader.rel_tot, hidden_size=230)
model_name = 'finetune'
framework.train(model, model_name)

