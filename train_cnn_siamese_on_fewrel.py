import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader import JSONFileDataLoader as DataLoader

max_length = 40
train_data_loader = DataLoader('./data/train_train.json', './data/glove.6B.50d.json', max_length=max_length)
val_data_loader = DataLoader('./data/train_val.json', './data/glove.6B.50d.json', max_length=max_length, rel2id=train_data_loader.rel2id, shuffle=False)

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.CNNSentenceEncoder(train_data_loader.word_vec_mat, max_length)
model = models.snowball.Siamese(sentence_encoder, hidden_size=230)

model_name = 'cnn_siamese_on_fewrel'

# set optimizer
batch_size = 128
train_iter = 30000

parameters_to_optimize = filter(lambda x:x.requires_grad, model.parameters())
optimizer = optim.SGD(parameters_to_optimize,
        1.,
        weight_decay=1e-5)

framework.train_siamese(model, model_name, optimizer=optimizer, batch_size=batch_size, train_iter=train_iter, learning_rate=1., warmup=True)
