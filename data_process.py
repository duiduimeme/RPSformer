# coding=gbk
import re

import joblib
import numpy as np
import pandas as pd
import os
from os import walk
import pickle
import collections

import torch
from torchtext import data
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vectors

SEED = 126
BATCH_SIZE = 2048
EMBEDDING_DIM = 100       # 词向量维度
LEARNING_RATE = 1e-3      # 学习率

def Cartesian(list1,list2):
    res_list = []
    for str1 in list1:
        for str2 in list2:
            res_list.append(str1 + str2)
    return res_list

def transfer():
    gen = ['A', 'U', 'G', 'C']
    struc = ['S', 'H', 'M', 'I', 'B', 'X', 'E']
    res = Cartesian(gen, struc)
    dict={}
    for i in range(len(res)):
        arr = np.zeros(28,dtype= np.int)
        arr[i] = 1
        dict[res[i]]=arr
    return dict

dict = transfer()

def to_list(string):
    list = [string[i:i + 2] for i in range(0, len(string), 2)]
    return list

def seq_encoding(string):
    str_list = to_list(string)
    encoding = list(map(lambda x:dict[x], str_list))
    return np.stack(encoding,axis=0)

def get_len(seq):
    return len(seq)/2

def fun(text):
    text_list = re.findall(".{2}", str(text))
    new_text = " ".join(text_list)
    return new_text

def data_split():
    # train_frame = pd.read_csv('raw_data/train.csv',index_col=0)
    train_data = pd.read_csv('raw_data/test.csv',index_col=0)
    train_data['text'] = list(map(fun, train_data['text']))
    # train_data, test_data = train_test_split(train_frame, test_size=0.4, random_state=SEED, shuffle=True)
    # test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=SEED, shuffle=True)
    # train_data = train_test_split(train_frame, test_size=0, random_state=SEED, shuffle=True)
    print(len(train_data))
    print(train_data.head())
    # print(len(val_data))
    # print(len(test_data))
    # train_data.to_csv('data/test.csv')
    # val_data.to_csv('data/dev.csv')
    # test_data.to_csv('data/test.csv')

# 设置device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# data_split()
TEXT = data.Field(tokenize=lambda x:x.split(), batch_first=True, lower=True) # 是否Batch_first. 默认值: False.
LABEL = data.LabelField(dtype=torch.float)
def get_dataset(path,text_field,label_field):
    fields = [('text', text_field), ('label', label_field)]  # torchtext文本配对关系
    examples = []
    print("path:",'data/'+path)
    frame = pd.read_csv('data/'+path)
    text = frame['text']
    label = frame['label']
    print("label:",len(label))
    for i in range(len(text)):
        # print(text[i])
        # print(text[i].split())
        # print(label[i])
        examples.append(data.Example.fromlist([text[i], label[i]], fields=fields))
    return examples,fields

def get_data(path,text_field,label_field):
    fields = [('text', text_field), ('label', label_field)]  # torchtext文本配对关系
    examples = []
    print("path:",'data/'+path)
    frame = pd.read_csv('data/'+path)

    text = frame['text']
    label = frame['label']
    print("label:",len(label))
    for i in range(len(text)):
        examples.append(data.Example.fromlist([text[i], label[i]], fields=fields))
    frame = pd.read_csv('data/test.csv')
    text = frame['text']
    label = frame['label']
    for i in range(len(text)):
        # print(text[i])
        # print(text[i].split())
        # print(label[i])
        examples.append(data.Example.fromlist([text[i], label[i]], fields=fields))
    print(len(examples))
    return examples,fields
# 得到构建Dataset所需的examples 和 fields
train_examples, train_fileds = get_data('train.csv', TEXT, LABEL)
dev_examples, dev_fields = get_dataset('dev.csv', TEXT, LABEL)
test_examples, test_fields = get_dataset('test.csv', TEXT, LABEL)

# 构建Dataset数据集
train_data = data.Dataset(train_examples, train_fileds)
dev_data = data.Dataset(dev_examples, dev_fields)
test_data = data.Dataset(test_examples, test_fields)

vectors = Vectors(name='vector/word2vec100.vector')
TEXT.build_vocab(train_data, max_size=100, vectors=vectors)
LABEL.build_vocab(train_data)
# print(len(TEXT.vocab))
# print(TEXT.vocab.itos)
# print(len(test_data))
# for t in test_data:
#     print(t.text, t.label)

train_iterator, dev_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, dev_data, test_data),
    batch_size=BATCH_SIZE,
    device='cpu',
    sort=False)

