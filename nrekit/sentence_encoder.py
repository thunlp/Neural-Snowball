import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim
from . import network

from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

class CNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder(x)
        return x

class PCNNSentenceEncoder(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, hidden_size=230):
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = network.embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.encoder = network.encoder.Encoder(max_length, word_embedding_dim, pos_embedding_dim, hidden_size)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.encoder.pcnn(x, inputs['mask'])
        return x

class BERTSentenceEncoder(nn.Module):

    def __init__(self, pretrain_path): 
        nn.Module.__init__(self)
        self.bert = BertModel.from_pretrained(pretrain_path)

    def forward(self, inputs):
        x, _ = self.bert(inputs['word'])
        x = x[-1] # (batch_size, max_length, hidden_size)
        x = x[:, 0, :] # (batch_size, hidden_size)
        return x

