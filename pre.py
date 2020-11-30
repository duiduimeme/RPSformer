# coding=gbk
import torch
import  numpy as np
import AttfoldModel
from data_process import TEXT, test_iterator


# from learning import evaluate, criteon, binary_acc
def binary_acc(preds, y):
    preds = torch.round(torch.sigmoid(preds))
    correct = torch.eq(preds, y).float()
    acc = correct.sum() / len(correct)
    return acc
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AttfoldModel.MyTransformerModel(30,30, p_drop=0.5, h=2, output_size=1)
model.load_state_dict(torch.load("wordavg-model1.pt"))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
criteon = torch.nn.BCEWithLogitsLoss()
avg_loss = []
avg_acc = []
model.eval()  # 表示进入测试模式
for batch in test_iterator:
    # batch = torch.Tensor(batch).to(device)
    # print(batch.text.size())
    mask = torch.Tensor((~(batch.text == TEXT.vocab.stoi['<pad>'])).float())
    pred = model(batch.text, mask)
    # print(pred.size()==batch.label.size())
    loss = criteon(pred, batch.label)
    acc = binary_acc(pred, batch.label).item()
    avg_loss.append(loss.item())
    avg_acc.append(acc)

avg_loss = np.array(avg_loss).mean()
avg_acc = np.array(avg_acc).mean()
print(f'Test. Loss: {avg_loss:.3f} |  Test. Acc: {avg_acc * 100:.2f}%')
