import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 40
train_data_loader = JSONFileDataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/train_val.json', './data/glove.6B.50d.json', max_length=max_length)

framework = nrekit.framework.PretrainFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model = models.proto.Proto(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=None, hidden_size=230)
model_name = 'proto_encoder'
checkpoint = framework.__load_model__('./checkpoint/' + model_name + '.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()
framework.eval_encoder(model, eval_iter=2000, support=True)
