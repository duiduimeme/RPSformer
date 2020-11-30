# coding=gbk
import os
import random
import time

# import torchsnooper
from sklearn.metrics import roc_auc_score

import AttfoldModel
import torch
import numpy as np
from torchtext.vocab import Vectors

SEED = 126
BATCH_SIZE = 4096
EMBEDDING_DIM = 100       # 词向量维度
LEARNING_RATE = 1e-3      # 学习率
#为了保证实验结果可以复现，我们经常会把各种random seed固定在某一个值
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# 计算准确率
from torch.utils import data
import AttfoldModel
from data_process import TEXT,train_iterator, dev_iterator, test_iterator

# 设置device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# os.environ["CUDA_VISIBLE_DEVICES"] = '2'


def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc


# 训练函数
def train(model, iterator, optimizer, criteon):
    avg_loss = []
    avg_acc = []
    model.train()  # 表示进入训练模式
    print("train......")
    for i, batch in enumerate(iterator):
        mask = torch.Tensor(1-(batch.text == TEXT.vocab.stoi['<pad>']).float())
        # print("mask:",mask)
        pred = model(batch.text, mask)
        # print("pred:",pred)
        loss = criteon(pred, batch.label)
        # print("loss:",loss)
        acc = binary_acc(pred, batch.label).item()  # 计算每个batch的准确率
        # print("第"+i+"个epoch训练的acc:",acc)
        avg_loss.append(loss.item())
        avg_acc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    avg_acc = np.array(avg_acc).mean()
    avg_loss = np.array(avg_loss).mean()
    return avg_loss, avg_acc


# 评估函数
def evaluate(model, iterator, criteon):
    avg_loss = []
    avg_acc = []
    model.eval()  # 表示进入测试模式
    res = []
    pre_res = []
    with torch.no_grad():
        for batch in iterator:
            # batch = torch.Tensor(batch).to(device)
            mask = torch.Tensor(1 - (batch.text == TEXT.vocab.stoi['<pad>']).float())
            pred = model(batch.text, mask)
            res.extend(batch.label)
            pre_res.extend(pred)
            loss = criteon(pred, batch.label)
            acc = binary_acc(pred, batch.label).item()
            avg_loss.append(loss.item())
            avg_acc.append(acc)
    auc = roc_auc_score(res,pre_res)
    avg_loss = np.array(avg_loss).mean()
    avg_acc = np.array(avg_acc).mean()
    return avg_loss, avg_acc,auc


drop = 0.7
file_name = 'wordavg-model5.pt'
model = AttfoldModel.MyTransformerModel(30, EMBEDDING_DIM, p_drop=drop, h=2, output_size=1)
pretrained_embedding = TEXT.vocab.vectors
model.embeddings.embed.weight.data.copy_(pretrained_embedding)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
criteon = torch.nn.BCEWithLogitsLoss()
# 训练模型，并打印模型的表现
best_valid_acc = float('-inf')
print("build model")
for epoch in range(30):
    print("epoch:",epoch)
    start_time = time.time()
    train_loss, train_acc = train(model, train_iterator, optimizer, criteon)
    dev_loss, dev_acc,auc = evaluate(model, dev_iterator, criteon)
    end_time = time.time()
    epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
    if dev_acc > best_valid_acc:  # 只要模型效果变好，就保存
        best_valid_acc = dev_acc
        torch.save(model.state_dict(), file_name)
    print(f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs:.2f}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
    print(f'\t Val. Loss: {dev_loss:.3f} |  Val. Acc: {dev_acc * 100:.2f}%',auc)

# 用保存的模型参数预测数据
print("预测集预测。。。。。")
model.load_state_dict(torch.load(file_name))
test_loss, test_acc = evaluate(model, test_iterator, criteon)
print(file_name+f':Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc * 100:.2f}%',auc)
print("drop:",drop)