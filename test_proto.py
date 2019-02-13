import models
import nrekit
import sys
import torch
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
model = models.proto.Proto(sentence_encoder, base_class=train_data_loader.rel_tot, siamese_model=model2, hidden_size=230)

# load pretrain
checkpoint = torch.load('./checkpoint/proto_encoder.pth.tar')['state_dict']
checkpoint2 = torch.load('./checkpoint/cnn_siamese.pth.tar')['state_dict']
for key in checkpoint2:
    checkpoint['siamese_model.' + key] = checkpoint2[key]
model.load_state_dict(checkpoint)
model.cuda()
model_name = 'proto'

# eval
# framework.train(model, model_name, model2=model2)
# framework.eval_siamese(model2, threshold=0.99)
# print('')
model.get_protos(train_data_loader)
framework.eval(model, threshold=0, threshold_for_snowball=0.9, support_size=10)
