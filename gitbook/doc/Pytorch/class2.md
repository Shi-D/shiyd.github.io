[TOC]

# 模型训练

* 模型一般需要训练若干epoch

* 每个epoch我们都把所有的数据分成若干个batch

* 把每个batch的输入和输出都包装成cuda tensor

* forward pass，通过输入的句子预测每个单词的下一个单词

* 用模型的预测和正确的下一个单词计算cross entropy loss

* 清空模型当前的gradient

* backward pass

* 更新模型参数

* 每隔一定的iteration输出模型在当前iteration的loss，以及在验证数据集上做模型评估

  

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud

from collections import Counter
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

CUDA_EXISTANCE = torch.cuda.is_available()

random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
if CUDA_EXISTANCE:
    torch.manual_seed(1)


C = 3 #context window
K = 100 #negative sampling
NUM_EPOCH = 1
MAX_VOCAB_SIZE = 100
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100


def word_tokenize(text):
    return text.split()


with open("text8.train.txt", 'r') as fin:
    text = fin.read()

text = text.split()

vocab = dict(Counter(text).most_common(MAX_VOCAB_SIZE-1))
vocab["<UNK>"] = len(text) - np.sum(list(vocab.values()))
print(vocab)

idx_to_word = [word for word in vocab.keys()]
word_to_idx = {word:i for i, word in enumerate(idx_to_word)}

word_counts = np.array([count for count in vocab.values()], dtype=np.float32)
word_freqs = word_counts / np.sum(word_counts)
word_freqs = word_freqs ** (3./4)
word_freqs = word_freqs / np.sum(word_freqs) #normalizen
VOCAB_SIZE = len(idx_to_word)

class WordEmbeddingDataset(tud.Dataset):
    def __init__(self, text, word_to_idx, idx_to_word, word_freqs, word_counts):
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = [word_to_idx.get(word, word_to_idx['<UNK>']) for word in text] # dict.get(key, default=None)返回指定键的值，若不在字典中返回默认值
        self.text_encoded = torch.LongTensor(self.text_encoded)
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)

    def __len__(self):
        #这个数据集有多少item
        return len(self.text_encoded)

    def __getitem__(self, idx):
        center_word = self.text_encoded[idx]
        pos_indices = list(range(idx-C, idx)) + list(range(idx+1, idx+C+1)) # window内单词
        pos_indices = [(i+len(self.text_encoded)) % len(self.text_encoded) for i in pos_indices] # 取余，防止超出text长度
        pos_words = self.text_encoded[pos_indices] # 周围单词
        neg_words = torch.multinomial(self.word_freqs, K * pos_words.shape[0], True) # 负采样单词
        return center_word, pos_words, neg_words


dataset = WordEmbeddingDataset(text, word_to_idx, idx_to_word, word_freqs, word_counts)
print('idx_to_word', dataset.idx_to_word[:100])
print('word_to_idx', dataset.word_to_idx)
print('word_freqs', dataset.word_freqs)
print('word_counts', dataset.word_counts)
print('text_encoded', dataset.text_encoded)


dataloader = tud.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for i, (center_word, pos_words, neg_words) in enumerate(dataloader):
    print('i', i)
    print('---------dataloader--------------\n-------center_word-------\n', center_word,'\n-------pos_words-------\n', pos_words,'\n-------neg_word-------\n', neg_words)
    break


'''定义pytorch模型'''
class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)

    def forward(self, input_labels, pos_labels, neg_labels):
        '''
        input_labels:[batch_size, embed_size] 即embed_size = 100
        pos_labels:[batch_size, (window_size * 2), embed_size]
        neg_labels:[batch_size, (window_size * 2 * K), embed_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size] 即embed_size = 100
        pos_embedding = self.out_embed(pos_labels) # [batch_size, (window_size * 2), embed_size]
        neg_embedding = self.in_embed(neg_labels) # [batch_size, (window_size * 2 * K), embed_size]

        input_embedding = input_embedding.unsqueeze(2) #input_embedding变为[batch_size, embed_size, 1]
        pos_dot = torch.bmm(pos_embedding, input_embedding).squeeze(2) #pos_dot变为[batch_size, (window_size * 2), 1]  =>  squeezy后变为[batch_size, (window_size * 2)]
        neg_dot = torch.bmm(neg_embedding, input_embedding).squeeze(2) #pos_dot变为[batch_size, (window_size * 2 * k), 1]  =>  squeezy后变为[batch_size, (window_size * 2 * k)]

        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(neg_dot).sum(1)

        loss = log_neg + log_pos

        return -loss

    def input_embedding(self):
        return self.input_embed.weight.data.cpu().numpy()


model = EmbeddingModel(VOCAB_SIZE, EMBEDDING_SIZE)
if CUDA_EXISTANCE:
    model = model.cuda()

optimizer = torch.optim.SGD(model.parameters(), lr = LEARNING_RATE)


loss_list = []
epoch_list = []

for epoch in range(NUM_EPOCH):
    for i, (input_labels, pos_labels, neg_labels) in enumerate(dataloader):
        input_labels = input_labels.long()
        pos_labels = pos_labels.long()
        neg_labels = neg_labels.long()
        if CUDA_EXISTANCE:
            input_labels = input_labels.cuda()
            pos_labels = pos_labels.cuda()
            neg_labels = neg_labels.cuda()

        optimizer.zero_grad()

        loss = model(input_labels, pos_labels, neg_labels).mean()
        loss.backward()

        optimizer.step()


        if i%200 == 0:
            print('epoch', epoch, 'iteration', i, 'loss', loss.item())
            loss_list.append(loss.item())
            epoch_list.append(i)


plt.plot(epoch_list, loss_list, marker='+')

plt.show()


```

Loss 

```python
epoch 0 iteration 0 loss 1728.7784423828125
epoch 0 iteration 200 loss 0.5566979646682739
epoch 0 iteration 400 loss 0.28769049048423767
epoch 0 iteration 600 loss 0.03424171730875969
epoch 0 iteration 800 loss 0.23394018411636353
epoch 0 iteration 1000 loss 0.02253451757133007
epoch 0 iteration 1200 loss 0.1249169111251831
epoch 0 iteration 1400 loss 0.5794429183006287
epoch 0 iteration 1600 loss 0.22607421875
epoch 0 iteration 1800 loss 0.15592460334300995
epoch 0 iteration 2000 loss 0.009277193807065487
epoch 0 iteration 2200 loss 0.041427891701459885
epoch 0 iteration 2400 loss 0.16730476915836334
epoch 0 iteration 2600 loss 0.04706058278679848
epoch 0 iteration 2800 loss 0.055349815636873245
epoch 0 iteration 3000 loss 0.09956461191177368
epoch 0 iteration 3200 loss 0.06921997666358948
epoch 0 iteration 3400 loss 0.008648096583783627
epoch 0 iteration 3600 loss 0.003158848499879241
epoch 0 iteration 3800 loss 0.0010388040682300925
epoch 0 iteration 4000 loss 0.030211854726076126
epoch 0 iteration 4200 loss 0.0043488554656505585
epoch 0 iteration 4400 loss 0.015924423933029175
epoch 0 iteration 4600 loss 0.00010472029680386186
epoch 0 iteration 4800 loss 0.14710034430027008
epoch 0 iteration 5000 loss 0.0026014982722699642
```

