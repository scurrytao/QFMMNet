# 对比试验TMC

import scipy.io as io
import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import xlrd

from Networks_FMM_9 import HGNN
# from efmm import FuzzyMinMaxNN
from FMinMax import FuzzyMinMaxNN
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.utils.data as utils
from xlutils.copy import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
def sensitivity(output, labels):
    preds = output.max(1)[1].type_as(labels)
    acc_mask = preds * labels
    return acc_mask.sum(0).double() / labels.sum(0).item()

def specially(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc_mask = preds * labels
    c1 = acc_mask.sum(0).double()
    c2 = correct - c1
    c3 = len(labels)-labels.sum(0).item()
    c4 = c2/c3
    return c4

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=2000, help='Number of epochs to train.')
parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N', help='gradually increase the value of lambda from 0 to 1')
parser.add_argument('--lr', type=float, default=5e-5, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-1, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=250, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args = parser.parse_args()
args.device = 'gpu'
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# Load data

data1 = io.loadmat('D:\Multi-view\DFC_Data\Huaxi_311\DFC_Huaxi_50_15.mat')
x=1
data3 = io.loadmat('D:\Fuzzyminmax\VAFNNs_Huaxi\Data_all_one_1\Data_all_idx_rand_'+str(x)+'.mat')

x_DFC = data1["Data"]

x_DFC = x_DFC.astype(np.float32)

idx_train = data3["train_idx"]
# idx_val = data3["val_idx"]
idx_test = data3["test_idx"]
idx_train = idx_train.astype(np.float32)
# idx_val = idx_val.astype(np.float32)
idx_test = idx_test.astype(np.float32)
A = x_DFC.shape
winNum = A[3]
y = data1["Y"]
y = np.transpose(y,(1,0))
x1_DFC = torch.FloatTensor (x_DFC)

y1 = torch.LongTensor(y)
idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
y1 = Variable (y1)
x1_DFC = Variable (x1_DFC)

labels = y1
x1_DFC = torch.unsqueeze(x1_DFC, 1)

labels = torch.squeeze(labels)
idx_train = torch.squeeze(idx_train)
idx_test = torch.squeeze(idx_test)
# idx_val = torch.squeeze(idx_val)
idx_train = torch.LongTensor(idx_train)
# idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)
fea_train_DFC = x1_DFC[idx_train,:,:,:,:]
# fea_val_DFC = x1_DFC[idx_val
#
# ,:,:,:,:]
fea_test_DFC = x1_DFC[idx_test,:,:,:,:]

train_dataset = utils.TensorDataset(fea_train_DFC, labels[idx_train])
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, drop_last=True)
# test_dataset = utils.TensorDataset(fea_test_DFC, labels[idx_test])
# test_loader = DataLoader( test_dataset, batch_size=48, shuffle=True, drop_last=False)
# Model and optimizer
model = HGNN(classifier_dims=[[2], [2], [2], [2], [2], [2], [2], [2], [2]],
             views=9,
             dim_inner=1,
             dim_out=64,
             dim_out_DF=16,
             nhid1=64,
             nclass=2,
             winNum=winNum,
             lambda_epochs=args.lambda_epochs)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr,
                       weight_decay=args.weight_decay)

fea_train_DFC, labels = Variable(fea_train_DFC), Variable(labels)
model = model.to(args.device)
labels = labels.to(args.device)

def train(epoch):
    t = time.time()
    model.train()
    f = {}
    Fmin = 0
    Fmax = 0
    for i, (DFC, target) in enumerate(train_loader):
        DFC = DFC.to(args.device)
        target = target.to(args.device)
        fuzzyy, new, pre, loss, evidence, Fmin, Fmax = model(f, DFC, target, epoch, Fmin, Fmax)
        acc_train = accuracy(new, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss.data.item()),
    #       'acc_train: {:.4f}'.format(acc_train.data.item()))
    return fuzzyy, evidence, target, Fmin, Fmax

def compute_test(epoch, fuzzyy, Fmin, Fmax):
    model.eval()
    loss_meter = AverageMeter()
    TP, TN, FP, FN = 0, 0, 0, 0
    correct_num = 0
    # for i, (DFC, target) in enumerate(test_loader):
    DFC = fea_test_DFC.to(args.device)
    DFC = DFC.to(args.device)
    with torch.no_grad():
        target = Variable(labels[idx_test].long().cuda())

        new, pre, loss, evidence = model(fuzzyy, DFC, target, epoch, Fmin, Fmax)
        _, predicted = torch.max(new, 1)    # _代表不需要用到的变量（具体的value，也就是输出的最大值）
        # correct_num
        correct_num += (predicted == target).sum().item()
        # TP
        TP += (predicted*target == 1).sum().item()
        # TN
        TN = correct_num-TP
        # FP
        FP += (predicted-target == 1).sum().item()
        # FN
        FN += (predicted-target == -1).sum().item()
        loss_meter.update(loss.item())

    # acc
    if (TP + TN + FP + FN) == 0:
        acc = 0
    else:
        acc = (TP + TN) / (TP + TN + FP + FN)

    # 阳性预测值（positive predict，PPV）指预测出的全部阳性样本中，真阳性所占的比例。表示判定为阳性类样本中，有多大概率预测是正确的。
    if (TP + FP) == 0:
        PPV = 0
    else:
        PPV = TP / (TP + FP)

    # 阴性预测值（negative predict，NPV）指预测出的全部阴性样本中，真阴性所占的比例。它表示判定为阴性类样本中，有多大概率预测是正确的。
    if (TN + FN) == 0:
        NPV = 0
    else:
        NPV = TN / (TN + FN)

    # Recall(召回率，又称为Sensitivity(敏感性))
    if (TP + FN) == 0:
        sen = 0
    else:
        sen = TP / (TP + FN)

    # spe
    if (TN + FP) == 0:
        spe = 0
    else:
        spe = TN / (TN + FP)

    # F1
    if (PPV + sen) == 0:
        F_1 = 0
    else:
        F_1 = (2 * PPV * sen) / (PPV + sen)

    return acc, sen, spe, PPV, NPV, F_1, evidence

def append_to_excel(words, filename):
    '''
    追加数据到excel
    :param words: 【item】 [{},{}]格式
    :param filename: 文件名
    :return:
    '''
    try:
        # 打开excel
        word_book = xlrd.open_workbook(filename)
        # 获取所有的sheet表单。
        sheets = word_book.sheet_names()
        # 获取第一个表单
        work_sheet = word_book.sheet_by_name(sheets[0])
        # 获取已经写入的行数
        old_rows = work_sheet.nrows
        # 获取表头信息
        heads = work_sheet.row_values(0)
        # 将xlrd对象变成xlwt
        new_work_book = copy(word_book)
        # 添加内容
        new_sheet = new_work_book.get_sheet(0)
        i = old_rows
        for item in words:
            for j in range(len(heads)):
                new_sheet.write(i, j, item[heads[j]])
            i += 1
        new_work_book.save(filename)
        print('追加成功！')
    except Exception as e:
        print('追加失败！', e)

def showfigs(fuzzy, test, test_labels, epoch, v):
    def draw_box(ax, a, b, color):
        width = abs(a[0] - b[0])
        height = abs(a[1] - b[1])
        ax.add_patch(patches.Rectangle(a, width, height, fill=False, edgecolor=color))

    """
        plot dataset
    """
    fig1 = plt.figure()
    ax = fig1.add_subplot(111, aspect='equal', alpha=0.7)

    """
        plot Hyperboxes
    """
    for i in range(len(fuzzy.V)):
        if fuzzy.hyperbox_class[i] == [1]:
            draw_box(ax, fuzzy.V[i][:2], fuzzy.W[i][:2], color='#8CC051')
        elif fuzzy.hyperbox_class[i] == [2]:
            draw_box(ax, fuzzy.V[i][:2], fuzzy.W[i][:2], color='#B26F66')
        else:
            draw_box(ax, fuzzy.V[i][:2], fuzzy.W[i][:2], color='b')

    for i in range(len(test)):
        if test_labels[i] == [1]:
            ax.scatter(test[i][0], test[i][1], marker='^', c='#8CC051')
        elif test_labels[i] == [2]:
            ax.scatter(test[i][0], test[i][1], marker='*', c='#B26F66')
        else:
            ax.scatter(test[i][0], test[i][1], marker='o', c='b')

    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.title('Hyperboxes created during training')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    save_path = 'D:\hbfigs_train'
    save_file = os.path.join(save_path, 'figure'+str(epoch)+'_'+str(v)+'.png')
    plt.savefig(save_file, format='png', dpi=1200)
    plt.show()
    return 0


best_acc = 0
sen, spe, PPV, NPV, F_1 = 0, 0, 0, 0, 0
best_epoch = 0
for epoch in range(args.epochs):
    fuzzyy, evidence, target, Fmin, Fmax = train(epoch)
    # if epoch > 500:
    #     for v in range(len(fuzzyy)):
    #         detached_tensor = evidence[v].cpu().detach().numpy()
    #         target_numpy = target.cpu().detach().numpy()
    #         detached_tensor_min = np.min(detached_tensor, axis=0)
    #         detached_tensor_max = np.max(detached_tensor, axis=0)
    #         detached_tensor = (detached_tensor - detached_tensor_min) / (detached_tensor_max - detached_tensor_min)
    #         target_numpy = target_numpy + 1
    #         target_numpy = target_numpy.reshape(-1, 1)
    #         detached_tensor, target_numpy = detached_tensor.tolist(), target_numpy.tolist()
    #         showfigs(fuzzyy[v], detached_tensor, target_numpy, epoch, v)
    # print(model)
    # print(compute_test(epoch, fuzzyy))
    acc_test, sen_test, spe_test, PPV_test, NPV_test, F_test, evidence = compute_test(epoch, fuzzyy, Fmin, Fmax)
    if acc_test > best_acc:
    # if epoch%50 == 0:
        best_acc = acc_test
        sen, spe, PPV, NPV, F_1 = sen_test, spe_test, PPV_test, NPV_test, F_test
        best_epoch = epoch + 1
        print("acc:", format(best_acc*100, '.2f'), "    best_epoch:", best_epoch)
        print("sen:", format(sen*100, '.2f'), "    spe:", format(spe*100, '.2f'), "   PPV:", format(PPV*100, '.2f'), "   NPV:", format(NPV*100, '.2f'), "   F1:", format(F_1*100, '.2f'))
        # words = [{'epoch': best_epoch, 'acc': format(best_acc * 100, '.2f'), 'sen': sen * 100, 'spe': spe * 100,
        #       'ppv': PPV * 100, 'npv': NPV * 100, 'f1': F_1 * 100}]
        # # 追加内容
        # append_to_excel(words=words, filename='data'+str(x)+'.xls')
        # best_acc = 0.85
        # torch.save(model.state_dict(), '{}.pth'.format(epoch))
        # for v in range(len(fuzzyy)):
        #     detached_tensor = evidence[v].cpu().detach().numpy()
        #     target_numpy = labels[idx_test].cpu().detach().numpy()
        #     detached_tensor_min = Fmin[v]
        #     detached_tensor_max = Fmax[v]
        #     detached_tensor = (detached_tensor - detached_tensor_min) / (detached_tensor_max - detached_tensor_min)
        #     target_numpy = target_numpy + 1
        #     target_numpy = target_numpy.reshape(-1, 1)
        #     detached_tensor, target_numpy = detached_tensor.tolist(), target_numpy.tolist()
        #     showfigs(fuzzyy[v], detached_tensor, target_numpy, epoch, v)
print(best_epoch)
print(best_acc)



