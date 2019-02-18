import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 40
train_data_loader = JSONFileDataLoader('./data/val_support.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/val_query.json', './data/glove.6B.50d.json', max_length=max_length)

framework = nrekit.framework.PretrainFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model = models.siamese.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=None, hidden_size=230)
model_name = 'test10way_finetune'
checkpoint = framework.__load_model__('./checkpoint/cnn_encoder.pth.tar')
own_state = model.state_dict()
for name, param in checkpoint.items():
    if 'sentence_encoder' in own_state:
        onw_state[name].copy_(param)
        print('load ' + name)
framework.train_encoder(model, model_name, learning_rate=0.1)
