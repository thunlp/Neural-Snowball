import sys
sys.path.append('..')
import nrekit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Siamese(nn.Module):

    def __init__(self, sentence_encoder, hidden_size=230):
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder # Should be different from main sentence encoder
        self.fc = nn.Linear(hidden_size, 1)
        self.cost = nn.BCELoss(reduction="none")

    def forward(self, data):
        xdata, ydata = data
        x = self.sentence_encoder(xdata)
        y = self.sentence_encoder(ydata)
        dis = torch.pow(x - y, 2)
        score = F.sigmoid(self.fc(dis).squeeze())
        loss = self.cost(score, xdata['label'].float()).mean()
        pred = torch.zeros((score.size(0))).long().cuda()
        pred[score > 0.5] = 1
        acc = torch.mean((pred.view(-1) == xdata['label'].view(-1))).type(torch.FloatTensor)

class Finetune(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, base_class, hidden_size=230):
        nrekit.framework.Model.__init__(self, sentence_encoder)
        self.hidden_size = hidden_size
        self.base_class = base_class
        self.fc = nn.Linear(hidden_size, base_class)
        self.drop = nn.Dropout()

    def forward_base(self, data):
        x = self.sentence_encoder(data) # (batch_size, hidden_size)
        x = self.fc(x) # (batch_size, base_class)
        x = F.sigmoid(x)
        label = torch.zeros((batch_size, self.base_class)).long().cuda()
        label.scatter_(dim=1, src=1, index=data['label']) # (batch_size, base_class)
        
        self._loss = self.__loss__(x, label)
        _, pred = x.max(-1)
        self._accuracy = self.__accuracy__(pred, data['label'])

    def forward_new(self, data):
        support, query = data
        new_W = Variable(torch.zeros((hidden_size)))
        nn.init.xavier_normal(new_W)
        new_bias = Variable(torch.zeros((1)))
        optimizer = optim.Adam([new_W, new_bias], learning_rate=1e-2, weight_decay=0)

        # Finetune
        support_x = self.sentence_encoder(support) # (batch_size, hidden_size)
        for i in range(10):
            x = torch.matmul(support_x, new_W) + new_bias # (batch_size, 1)
            x = F.sigmoid(x)
            iter_loss = self.__loss__(x, support['label'])
            iter_loss.backward()
            optimizer.step()
        
        # Test
        query_x = self.sentence_encoder(query) # (batch_size, hidden_size)
        x = torch.matmul(query_x, new_W) + new_bias # (batch_size, 1)
        x = F.sigmoid(x)
        self._loss = self.__loss__(x, query['label'])
        pred = torch.zeros((x.size(0))).long().cuda()
        pred[x > 0.5] = 1
        self._accuracy = self.__accuracy__(pred, query['label'])
    
    
