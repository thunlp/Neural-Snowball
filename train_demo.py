import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)
train_distant = JSONFileDataLoader('./data/train_distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)
val_distant = JSONFileDataLoader('./data/val_distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)
test_distant = JSONFileDataLoader('./data/test_distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_data_loader, val_data_loader, test_data_loader, train_distant, val_distant, test_distant)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
sentence_encoder2 = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model2 = models.siamese.Siamese(sentence_encoder2, hidden_size=230)
model = models.siamese.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=model2, hidden_size=230)
model_name = 'siamese'
# framework.train(model, model_name)
framework.train(model, model_name, model2=model2)

