# coding: UTF-8
'''
Created on Jan 25, 2022
@author: Xingchen Li
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """Configuration parameters"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'   # The training set
        self.dev_path = dataset + '/data/dev.txt'       # Validation set
        self.test_path = dataset + '/data/test.txt'     # The test set
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', 
                            encoding='utf-8').readlines()]     # Category list
        self.vocab_path = dataset + '/data/vocab.pkl'          # vocabulary
        self.save_path = dataset + '/saved_dict/' + self.model_name + ".ckpt" # Model training results
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # Pre-train word vectors
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # equipment

        self.dropout = 0.5                      # Random inactivation
        self.require_improvement = 2000         # If the effect is not improved after 1000batch, the training will be finished in advance
        self.num_classes = len(self.class_list) # Number of categories
        self.n_vocab = 0                        # Glossary size, assigned at run time
        self.num_epochs = 100                   # Number of epoch
        self.batch_size = 128                   # mini-batch value
        self.pad_size = 32                      # Length of each sentence (fill in short and cut long)
        self.learning_rate = 1e-3               # Initial learning rate (learning exponential decay)
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300  # Word vector dimension
        self.filter_sizes = (2, 3, 4, 5)        # Convolution kernel size
        self.num_filters = 256                  # Number of convolution nuclei (Number of Channels)


'''Convolutional Neural Networks for Sentence Classification'''
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = conv(x)         # [128, 256, <31/30/29>, 1]
        x = F.relu(x)
        # Compress the last dimension
        x = x.squeeze(3)    # [128, 256, <31/30/29>]
        x = F.max_pool1d(x, x.size(2)) #[128, 256, 1]
        x = x.squeeze(2)    # [128, 256]
        return x

    def forward(self, x):
        # x = (batch_setence, batch_length)
        out = self.embedding(x[0])  # [128, 32, 300]
        out = out.unsqueeze(1)      # [128, 1, 32, 300] Added a channel dimension
        out = [self.conv_and_pool(out, conv) for conv in self.convs]
        # [[128, 256], [128, 256], [128, 256]]
        out = torch.cat(out, 1)     # [128, 768]
        out = self.dropout(out)
        out = self.fc(out)          # [128, 5]
        return out
