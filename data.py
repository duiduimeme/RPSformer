import joblib
import numpy as np
import pandas as pd
import os
from os import walk
import pickle
import collections
from sklearn.model_selection import train_test_split

dataset = 'rnastralign'
datasets = ['train.csv','test.csv']
seed = 8

data_path = '../raw_data/'
data_file_list = list()

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

train_frame = pd.read_csv('raw_data/train.csv')
seq_len_list = list(map(get_len, train_frame['text']))
train_encoding_list = list(map(seq_encoding, train_frame['text']))
print(len(train_encoding_list))
train_label = np.stack(train_frame['label'])

RNA_code = collections.namedtuple('RNA_code',
                                  'seq label length')
RNA_code_list = list()

for i in range(len(train_encoding_list)):
    RNA_code_list.append(RNA_code(seq=train_encoding_list[i],
                                  label=train_label[i],
                                  length=seq_len_list[i]))
# # training test split ги0.8 train   0.1 val   0.1 test)
def get_data():

    RNA_code_train, RNA_code_test = train_test_split(RNA_code_list,
                                                         test_size=0.4, random_state=seed)
    RNA_code_test, RNA_code_val = train_test_split(RNA_code_test,
                                                       test_size=0.5, random_state=seed)
    return RNA_code_train,RNA_code_val,RNA_code_test


# savepath = dataset+'_512'
#
# for i in ['train','test','val']:
#     with open(savepath+'/%s.pickle' % i,'wb') as f:
#         joblib.dump(eval('RNA_code_'+i),f)
