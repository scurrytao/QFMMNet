from torch import nn
import torch.nn.functional as F
import torch
from efmm import FuzzyMinMaxNN
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class HGNN(nn.Module):
    def __init__(self, classifier_dims, views, dim_inner, dim_out, dim_out_DF, nhid1, nclass, winNum, lambda_epochs=1, dropout=0.2):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.cov_SF1 = nn.Conv2d(
            dim_inner,
            dim_out,
            kernel_size=[90, 1],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.cov_SF2 = nn.Conv2d(
            dim_inner,
            dim_out,
            kernel_size=[1, 90],
            stride=[1, 1],
            padding=[0, 0],
            bias=False,
        )
        self.cov_SF3= nn.Conv2d(
            dim_out,
            dim_out*2,
            kernel_size=[90, 1],
            stride=[1, 1],
            padding=[0, 0],
        )
        self.cov_DF1 = nn.Conv3d(
            dim_inner,
            dim_out_DF,
            kernel_size=[90, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.cov_DF2 = nn.Conv3d(
            dim_inner,
            dim_out_DF,
            kernel_size=[1, 90, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
            bias=False,
        )
        self.cov_DF3 = nn.Conv3d(
            dim_out_DF ,
            dim_out_DF*2,
            kernel_size=[90, 1, 1],
            stride=[1, 1, 1],
            padding=[0, 0, 0],
        )
        self.cov_S2D1 = nn.Conv2d(
            dim_out,
            dim_out_DF,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
        )
        self.cov_D2S1 = nn.Conv2d(
            dim_out_DF,
            dim_out,
            kernel_size=[1, 1],
            stride=[1, 1],
            padding=[0, 0],
        )
        self.cov_D2S2 = nn.Conv1d(
            dim_out_DF*2,
            dim_out*2,
            kernel_size=[1],
            stride=[1],
            padding=[0],
        )
        self.cov_SF5 = nn.Conv2d(
            dim_out,
            dim_out * 2,
            kernel_size=[90,1],
            stride=[1,1],
            padding=[0,0],
        )
        self.cov_SF6 = nn.Conv1d(
            dim_out*2,
            dim_out*4,
            kernel_size=[90],
            stride=[1],
            padding=[0],
        )
        self.cov_S2D2 = nn.Conv1d(
            dim_out*2 ,
            dim_out_DF*2,
            kernel_size=[1,1],
            stride=[1,1],
            padding=[0],
        )
        self.cov_DF4 = nn.Conv2d(
            dim_out_DF * 2,
            dim_out_DF * 4,
            kernel_size=[90,1],
            stride=[1,1],
            padding=[0,0],
        )
        self.pool1 = nn.AvgPool3d(
            kernel_size=[1, 1, winNum],
            stride=[1, 1, 1],
        )
        self.pool2 = nn.MaxPool2d(
            kernel_size=[1, winNum],
            stride=[1, 1],
        )
        self.lin1 = nn.Linear(dim_out_DF*4*winNum, nhid1)
        self.lin2 = nn.Linear(nhid1, 32)
        self.lin3 = nn.Linear(32, 2)
        self.views = views
        self.nclass = nclass
        self.lambda_epochs = lambda_epochs
        self.Classifiers = nn.ModuleList([Classifier(classifier_dims[i], self.nclass) for i in range(self.views)])
        self.intermediate_variable = {}

    def Get_DS_B_U(self, alpha):
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(self.views):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = self.nclass / S[v]
        return b, u

    def Quality(self, b, u):
        u_all = 0
        b_max = {}
        Q = {}
        for i in range(len(u)):
            u_all = u_all + u[i]
        for key, value in b.items():

            max_value = torch.max(value, dim=1, keepdim=True)[0]
            # 将最大值添加到新的字典中
            b_max[key] = max_value
        for i in range(len(u)):
            Q[i] = -(torch.log(u[i]) * self.views * torch.exp(b_max[i]) / (u_all))

        return Q

    def forward(self, fuzzyy, x_DFC, target, global_step, Fmin, Fmax):
        x_1_DFC = F.relu(self.cov_DF1(x_DFC))
        x_1_DFC = x_1_DFC.repeat(1, 1, 90, 1, 1)
        x_2_DFC = F.relu(self.cov_DF2(x_DFC))
        x_2_DFC = x_2_DFC.repeat(1, 1, 1, 90, 1)
        x_DFC = x_1_DFC + x_2_DFC
        x_DFC = F.dropout(x_DFC, self.dropout)    # E2E_DFC
        x_DFC = F.relu(self.cov_DF3(x_DFC))    # E3N
        x_DFC = F.dropout(x_DFC, self.dropout)
        x_DFC = torch.squeeze(x_DFC, 2)    # (120,32,90,10)
        x_DFC = F.relu(self.cov_DF4(x_DFC))   # E2G (120,64,1,10)
        x_DFC = torch.squeeze(x_DFC, 2)    # (120,64,10)
        x_DFC = F.dropout(x_DFC, self.dropout)


        view0 = x_DFC[:, :, 0]    # (120,64)
        view0 = self.lin2(view0)
        view0 = F.dropout(view0, self.dropout)
        view0 = self.lin3(view0)

        view1 = x_DFC[:, :, 1]
        view1 = self.lin2(view1)
        view1 = F.dropout(view1, self.dropout)
        view1 = self.lin3(view1)

        view2 = x_DFC[:, :, 2]
        view2 = self.lin2(view2)
        view2 = F.dropout(view2, self.dropout)
        view2 = self.lin3(view2)

        view3 = x_DFC[:, :, 3]
        view3 = self.lin2(view3)
        view3 = F.dropout(view3, self.dropout)
        view3 = self.lin3(view3)

        view4 = x_DFC[:, :, 4]
        view4 = self.lin2(view4)
        view4 = F.dropout(view4, self.dropout)
        view4 = self.lin3(view4)

        view5 = x_DFC[:, :, 5]
        view5 = self.lin2(view5)
        view5 = F.dropout(view5, self.dropout)
        view5 = self.lin3(view5)

        view6 = x_DFC[:, :, 6]
        view6 = self.lin2(view6)
        view6 = F.dropout(view6, self.dropout)
        view6 = self.lin3(view6)

        view7 = x_DFC[:, :, 7]
        view7 = self.lin2(view7)
        view7 = F.dropout(view7, self.dropout)
        view7 = self.lin3(view7)

        view8 = x_DFC[:, :, 8]
        view8 = self.lin2(view8)
        view8 = F.dropout(view8, self.dropout)
        view8 = self.lin3(view8)

        if x_DFC.requires_grad == True:
            X = {0: view0, 1: view1, 2: view2, 3: view3, 4: view4, 5: view5, 6: view6, 7: view7, 8: view8}
            # neww = {0: new0, 1: new1, 2: new2, 3: new3, 4: new4, 5: new5, 6: new6, 7: new7, 8: new8, 9: new9}
            evidence = self.infer(X)

            fuzzy0, pre0, new0, min0, max0 = FMM(evidence[0], target)
            fuzzy1, pre1, new1, min1, max1 = FMM(evidence[1], target)
            fuzzy2, pre2, new2, min2, max2 = FMM(evidence[2], target)
            fuzzy3, pre3, new3, min3, max3 = FMM(evidence[3], target)
            fuzzy4, pre4, new4, min4, max4 = FMM(evidence[4], target)
            fuzzy5, pre5, new5, min5, max5 = FMM(evidence[5], target)
            fuzzy6, pre6, new6, min6, max6 = FMM(evidence[6], target)
            fuzzy7, pre7, new7, min7, max7 = FMM(evidence[7], target)
            fuzzy8, pre8, new8, min8, max8 = FMM(evidence[8], target)

            fuzzyy = {0: fuzzy0, 1: fuzzy1, 2: fuzzy2, 3: fuzzy3, 4: fuzzy4, 5: fuzzy5, 6: fuzzy6, 7: fuzzy7, 8: fuzzy8}
            self.intermediate_variable = fuzzyy
            Fmin = {0: min0, 1: min1, 2: min2, 3: min3, 4: min4, 5: min5, 6: min6, 7: min7, 8: min8}
            Fmax = {0: max0, 1: max1, 2: max2, 3: max3, 4: max4, 5: max5, 6: max6, 7: max7, 8: max8}

            loss = 0
            CELOSSOUTPUT = nn.CrossEntropyLoss()
            alpha = dict()
            for v_num in range(len(X)):
                alpha[v_num] = evidence[v_num] + 1
                # loss += ce_loss(target, alpha[v_num], self.nclass, global_step, self.lambda_epochs)
                loss += CELOSSOUTPUT(evidence[v_num], target)

            belief, uncertain = self.Get_DS_B_U(alpha)

            Q = self.Quality(belief, uncertain)
            new = Q[0] * new0 + Q[1] * new1 + Q[2] * new2 + Q[3] * new3 + Q[4] * new4 + Q[5] * new5 + Q[6] * new6 + Q[7] * new7 + Q[8] * new8

            _, pre = torch.max(new, 1)
            # new =
            loss = torch.mean(loss)
            # print(new)
            return fuzzyy, new, pre, loss, evidence, Fmin, Fmax

        if x_DFC.requires_grad == False:

            X = {0: view0, 1: view1, 2: view2, 3: view3, 4: view4, 5: view5, 6: view6, 7: view7, 8: view8}
            evidence = self.infer(X)

            pre0, new0 = FMM_test(evidence[0], target, fuzzyy[0], Fmin[0], Fmax[0])
            pre1, new1 = FMM_test(evidence[1], target, fuzzyy[1], Fmin[1], Fmax[1])
            pre2, new2 = FMM_test(evidence[2], target, fuzzyy[2], Fmin[2], Fmax[2])
            pre3, new3 = FMM_test(evidence[3], target, fuzzyy[3], Fmin[3], Fmax[3])
            pre4, new4 = FMM_test(evidence[4], target, fuzzyy[4], Fmin[4], Fmax[4])
            pre5, new5 = FMM_test(evidence[5], target, fuzzyy[5], Fmin[5], Fmax[5])
            pre6, new6 = FMM_test(evidence[6], target, fuzzyy[6], Fmin[6], Fmax[6])
            pre7, new7 = FMM_test(evidence[7], target, fuzzyy[7], Fmin[7], Fmax[7])
            pre8, new8 = FMM_test(evidence[8], target, fuzzyy[8], Fmin[8], Fmax[8])


            loss = 0
            CELOSSOUTPUT  = nn.CrossEntropyLoss()
            alpha = dict()
            for v_num in range(len(X)):
                alpha[v_num] = evidence[v_num] + 1
                # loss += ce_loss(target, alpha[v_num], self.nclass, global_step, self.lambda_epochs)
                loss += CELOSSOUTPUT(evidence[v_num], target)
            belief, uncertain = self.Get_DS_B_U(alpha)
            Q = self.Quality(belief, uncertain)
            new = Q[0] * new0 + Q[1] * new1 + Q[2] * new2 + Q[3] * new3 + Q[4] * new4 + Q[5] * new5 + Q[6] * new6 + Q[7] * new7 + Q[8] * new8

            # new = uncertain[0] * new0 + uncertain[1] * new1 + uncertain[2] * new2 + uncertain[3] * new3 + uncertain[4] * new4 + uncertain[5] * new5 + uncertain[6] * new6 + uncertain[7] * new7 + uncertain[8] * new8 + uncertain[9] * new9
            _, pre = torch.max(new, 1)
            # new =
            loss = torch.mean(loss)
            # print(new)

            return new, pre, loss, evidence



    def infer(self, input):
        """
        :param input: Multi-view data
        :return: evidence of every view
        """
        evidence = dict()
        for v_num in range(self.views):
            evidence[v_num] = self.Classifiers[v_num](input[v_num])
        return evidence

def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl

def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)    
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)

    annealing_coef = min(1, global_step / annealing_step)

    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)

    return (A + B)

class Classifier(nn.Module):
    def __init__(self, classifier_dims, nclass):
        super(Classifier, self).__init__()
        self.num_layers = len(classifier_dims)
        self.fc = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.fc.append(nn.Linear(classifier_dims[i], classifier_dims[i+1]))
        self.fc.append(nn.Linear(classifier_dims[self.num_layers-1], nclass))
        self.fc.append(nn.Softplus())

    def forward(self, x):
        h = self.fc[0](x)
        for i in range(1, len(self.fc)):
            h = self.fc[i](h)
        return h

def FMM(t, l):
    # t:data l:label
    detached_tensor = t.cpu().detach().numpy()
    target_numpy = l.cpu().detach().numpy()
    detached_tensor_min = np.min(detached_tensor, axis=0)
    detached_tensor_max = np.max(detached_tensor, axis=0)
    detached_tensor = (detached_tensor - detached_tensor_min) / (detached_tensor_max - detached_tensor_min)
    target_numpy = target_numpy + 1
    target_numpy = target_numpy.reshape(-1, 1)
    detached_tensor, target_numpy = detached_tensor.tolist(), target_numpy.tolist()
    fuzzy = FuzzyMinMaxNN(1, theta=0.3)

    fuzzy.train(detached_tensor, target_numpy, 1)

    def get_class(x):
        mylist = []
        for i in range(len(fuzzy.V)):
            mylist.append([fuzzy.fuzzy_membership(x, fuzzy.V[i], fuzzy.W[i])])
        result = np.multiply(mylist, fuzzy.U)
        mylist = []
        for i in range(fuzzy.clasess):
            mylist.append(max(result[:, i]))

        return mylist, [mylist.index(max(mylist)) + 1]

    new_list = []
    pre = []

    for i in range(len(detached_tensor)):
        a, b = get_class(detached_tensor[i])
        new_list.append(a)
        pre.append(b)

    new_list = torch.tensor(new_list).cuda()
    pre = torch.tensor(pre).cuda()
    pre = pre - 1
    pre = torch.squeeze(pre)
    return fuzzy, pre, new_list, detached_tensor_min, detached_tensor_max

def FMM_test(t, l, fuzzy, Fmin, Fmax):
    # t:data l:label
    detached_tensor = t.cpu().detach().numpy()
    target_numpy = l.cpu().detach().numpy()
    detached_tensor_min = Fmin
    detached_tensor_max = Fmax
    detached_tensor = (detached_tensor - detached_tensor_min) / (detached_tensor_max - detached_tensor_min)

    target_numpy = target_numpy + 1
    target_numpy = target_numpy.reshape(-1, 1)
    detached_tensor, target_numpy = detached_tensor.tolist(), target_numpy.tolist()


    def get_class(x):
        mylist = []
        for i in range(len(fuzzy.V)):
            mylist.append([fuzzy.fuzzy_membership(x, fuzzy.V[i], fuzzy.W[i])])
        result = np.multiply(mylist, fuzzy.U)
        mylist = []
        for i in range(fuzzy.clasess):
            mylist.append(max(result[:, i]))

        return mylist, [mylist.index(max(mylist)) + 1]


    new_list = []
    pre = []

    for i in range(len(detached_tensor)):
        a, b = get_class(detached_tensor[i])
        new_list.append(a)
        pre.append(b)

    new_list = torch.tensor(new_list).cuda()
    pre = torch.tensor(pre).cuda()
    pre = pre - 1
    pre = torch.squeeze(pre)
    return pre, new_list


