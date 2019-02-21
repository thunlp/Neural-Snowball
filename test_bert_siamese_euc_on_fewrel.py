import models
import nrekit
import sys
from torch import optim
from nrekit.data_loader_bert import JSONFileDataLoaderBERT as DataLoader

from pytorch_pretrained_bert import BertAdam

max_length = 90
train_data_loader = DataLoader('./data/train_train.json', vocab='./data/bert_vocab.txt', max_length=max_length)
val_data_loader = DataLoader('./data/val.json', vocab='./data/bert_vocab.txt', max_length=max_length, rel2id=train_data_loader.rel2id, shuffle=False)

framework = nrekit.framework.SuperviseFramework(train_data_loader, val_data_loader)
sentence_encoder = nrekit.sentence_encoder.BERTSentenceEncoder('./data/bert-base-uncased')
model = models.snowball.Siamese(sentence_encoder, hidden_size=768, drop_rate=0.1)

model_name = 'bert_siamese_euc_on_fewrel'

checkpoint = framework.__load_model__('./checkpoint/' + model_name + '.pth.tar.bak')
model.load_state_dict(checkpoint['state_dict'])
model.cuda()

framework.eval_siamese(model, eval_iter=500, threshold=0.5, s_num_size=4, s_num_class=8)
print('')
framework.eval_siamese(model, eval_iter=500, threshold=0.7, s_num_size=4, s_num_class=8)
print('')
framework.eval_siamese(model, eval_iter=500, threshold=0.9, s_num_size=4, s_num_class=8)
print('')
