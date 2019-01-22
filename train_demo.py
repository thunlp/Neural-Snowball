import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val.json', './data/glove.6B.50d.json', max_length=max_length)
test_data_loader = JSONFileDataLoader('./data/test.json', './data/glove.6B.50d.json', max_length=max_length)
distant = JSONFileDataLoader('./data/distant.json', './data/glove.6B.50d.json', max_length=max_length, distant=True)

framework = nrekit.framework.Framework(train_data_loader, val_data_loader, test_data_loader, distant)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
sentence_encoder2 = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model2 = models.siamese.Siamese(sentence_encoder2, hidden_size=230)
model2.load_state_dict(torch.load('./checkpoint/cnn_siamese.pth.tar')['state_dict'])
model = models.siamese.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=model2, hidden_size=230)
model.load_state_dict(torch.load('./checkpoint/cnn_encoder.pth.tar')['state_dict'])
model_name = 'siamese'
# framework.train(model, model_name, model2=model2)
framework.eval(model2, is_model2=True, threshold=0.5)
framework.eval(model2, is_model2=True, threshold=0.7)
framework.eval(model2, is_model2=True, threshold=0.9)
framework.eval(model2, is_model2=True, threshold=0.95)
framework.eval(model, threshold=0.5, threshold_for_snowball=0.7)
