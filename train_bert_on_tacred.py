import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader

max_length = 128
train_data_loader = JSONFileDataLoader('./data/tacred_train.json', max_length=max_length)
val_data_loader = JSONFileDataLoader('./data/tacred_val.json', max_length=max_length, rel2id=train_data_loader.rel2id, shuffle=False)

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder()
model = models.snowball.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=None, hidden_size=768)
model_name = 'bert_encoder_on_tacred'
model.NA_label = 13
framework.train_encoder(model, model_name, batch_size=32, train_iter=120000, lr_step_size=60000)
