import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader

max_length = 128
train_data_loader = DataLoader('./data/tacred_train.json', vocab='./data/bert_vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/tacred_val.json', vocab='./data/bert_vocab.txt', max_length=max_length, rel2id=train_data_loader.rel2id, shuffle=False)

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
model = models.snowball.Snowball(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=None, hidden_size=768)
model_name = 'bert_encoder_on_tacred'
model.NA_label = 13
framework.train_encoder_epoch(model, model_name, batch_size=32, train_epoch=30, learning_rate=5e-5, weight_decay=0, lr_step_size=1000000000)
