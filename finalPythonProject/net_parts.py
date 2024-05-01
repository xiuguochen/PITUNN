# untrained physics-informed networks的组成代码
# created by: YSL
# created date: 2023.6
import torch
import torch.optim
import time
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def read_col(file_path, col):
    """用于读取csv文件的数据

    :param file_path: 文件地址（字符串）
    :param col: 欲读取数据的列名（字符串）
    :return: 返回行向量数据
    """
    data = pd.read_csv(file_path)
    coldata = data[col]
    coldata = np.array(coldata, dtype=np.float32)
    outdata = coldata.reshape((1, coldata.size))
    return outdata


def adjust_learning_rate(optimizer, epoch, start_lr):
    lr = start_lr * (0.1 ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class linear_net(nn.Module):
    def __init__(self, in_ch, out_ch, layer_num=2, hidden_ch=2000):
        super(linear_net, self).__init__()
        layer_set = ''
        for i in range(layer_num):
            layer_set = layer_set + r'nn.Linear(hidden_ch, hidden_ch),\
            nn.ReLU(inplace=True),'
        net_set = r'nn.Sequential(\
            nn.Linear(in_ch, hidden_ch),\
            nn.ReLU(inplace=True),' + layer_set + \
                  r'nn.Linear(hidden_ch, out_ch),\
        )'
        self.layer = eval(net_set)

    def forward(self, x):
        y = self.layer(x)
        return y


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, dowmsampling=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if dowmsampling:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=3, padding=1, stride=strides)
        else:
            self.conv3 = None
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.relu(self.conv1(X))
        Y = self.conv2(Y)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)
    pass


class ResNet(nn.Module):
    def __init__(self, out_ch):
        super(ResNet, self).__init__()
        b1 = nn.Sequential(nn.Conv1d(1, 64, kernel_size=5, stride=1, padding=2),
                           nn.ReLU(),
                           nn.MaxPool1d(kernel_size=2, stride=2))

        b2 = nn.Sequential(*self.__resnet_block(64, 64, 3, Fisrtblock=True))
        b3 = nn.Sequential(*self.__resnet_block(64, 128, 4))
        b4 = nn.Sequential(*self.__resnet_block(128, 256, 6))
        b5 = nn.Sequential(*self.__resnet_block(256, 256, 3))
        self.conv = nn.Sequential(b1, b2, b3, b4, b5, nn.AvgPool1d(2), nn.Flatten())
        self.fc = nn.Linear(3072, out_ch)

        pass

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)

        return x.unsqueeze(0)
        pass

    def __resnet_block(self, input_channels, num_channels, num_residuals, Fisrtblock=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not Fisrtblock:
                blk.append(Residual(input_channels, num_channels, dowmsampling=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fix = nn.Flatten(0, -1)
        self.fc = nn.Linear(400*hidden_size, output_size)

    def forward(self, x):
        x = x.squeeze(0).squeeze(0).unsqueeze(-1).unsqueeze(-1)
        signal_size = x.size(0)

        hidden = self.init_hidden(signal_size)

        out, hidden = self.rnn(x, hidden)
        out = self.fix(out)

        out = self.fc(out)

        return out.unsqueeze(0).unsqueeze(0)

    def init_hidden(self, signal_size):
        hidden = torch.zeros(self.num_layers, signal_size, self.hidden_size).to("cuda:0")

        return hidden


def stopCriterion(running_loss, loss_item, epoch, num2, iter_num, t):
    sum_10 = np.sum(running_loss[epoch * num2 + iter_num-5: epoch * num2 + iter_num])
    cri = np.abs(5*loss_item-sum_10)/sum_10
    return cri < t


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def simpleNCS(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
              layers=4, hidden=2000, num=2, num2=300, f_set=0.05, beta=0.01, gama=5, range1=0.2, range2=0.8, seed=5):
    """使用不考虑误差项的物理模型
    :param M_basis:
    """
    # 定义网络、损失函数、优化器
    series_num = int(M_basis.shape[2] * f_set)
    zero_col = torch.zeros([1, 1, M_basis.shape[2] - series_num]).to('cuda:0')
    sample_points = int(Y.shape[2])
    setup_seed(seed)
    myNet = linear_net(sample_points, 3 * series_num, layers, hidden).to("cuda:0")
    # myNet = ResNet(3 * series_num).to("cuda:0")
    # myNet = RNN(1, 400, 2, 3 * series_num).to("cuda:0")
    criterion = nn.MSELoss().to("cuda:0")
    start_lr = 0.01
    optimizer = torch.optim.Adam(myNet.parameters(), lr=start_lr)
    # 开始训练
    running_loss = np.zeros(num * num2)
    start = time.time()
    for epoch in range(num):
        adjust_learning_rate(optimizer, epoch, start_lr)
        stopFlag = False
        for iter_num in range(num2):
            optimizer.zero_grad()
            output_tensor = myNet(Y)
            temp1 = torch.cat([output_tensor[:1, :1, 0:series_num], zero_col], dim=2)
            temp2 = torch.cat([output_tensor[:1, :1, series_num:2 * series_num], zero_col], dim=2)
            temp3 = torch.cat([output_tensor[:1, :1, 2 * series_num:3 * series_num], zero_col], dim=2)

            N_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp1])
            C_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp2])
            S_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp3])

            I_generate = 2 - 2 * N_generate * cos_delta_2 + C_generate * cos_delta_1Minus2 \
                         - C_generate * cos_delta_1Plus2 \
                         + S_generate * sin_delta_1Minus2 - S_generate * sin_delta_1Plus2
            error = (Y - I_generate)

            loss = (torch.norm(error[:1, :1, int(range1 * sample_points):int(range2 * sample_points)], p=2, dim=2).squeeze(0)
                    + beta * torch.norm(output_tensor, p=1, dim=2).squeeze(0)
                    + gama * torch.norm(N_generate ** 2 + C_generate ** 2 +
                                        S_generate ** 2 - 1, p=2, dim=2).squeeze(0))

            loss.backward()
            optimizer.step()
            running_loss[epoch * num2 + iter_num] = loss.item()
            if epoch * num2 + iter_num > 20:
                if stopCriterion(running_loss, loss.item(), epoch, num2, iter_num, t=0.005):
                    stopFlag = True
                    break
        if stopFlag:
            # print('11111')
            break
        #     if epoch * num2 + iter_num > 0:
        #         out_N_simple = N_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         out_C_simple = C_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         out_S_simple = S_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         x = pd.DataFrame({'out_N_simple': out_N_simple, 'out_C_simple': out_C_simple,
        #                           'out_S_simple': out_S_simple})
        #         x.to_csv(r"C:\Users\98072\ysl_file\PTUNN_code\earlyStopping\net1data-0.1\NCSsimple-"+
        #                  str(epoch * num2 + iter_num) + "-" + str(loss.item()) + ".csv")
            # print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')

    print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')
    end = time.time()
    time_cost = end - start
    print('time cost: ', time_cost, 's \n')

    return N_generate, C_generate, S_generate, I_generate, running_loss, time_cost


def epsilonNCS(Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
               cos_delta_1, sin_delta_1,
               N, C, S,
               layers=4, hidden=2000, num=2, num2=300, range1=0.2, range2=0.8, seed = 5):
    """使用不考虑误差项的物理模型
    :param M_basis:
    """
    # 定义网络、损失函数、优化器
    sample_points = int(Y.shape[2])
    setup_seed(seed)
    myNet = linear_net(sample_points, 6, layers, hidden).to("cuda:0")
    # myNet = ResNet(6).to("cuda:0")
    # myNet = RNN(1, 400, 2, 6).to("cuda:0")
    start_lr = 0.01
    optimizer = torch.optim.Adam(myNet.parameters(), lr=start_lr)
    # 开始训练
    running_loss = np.zeros(num * num2)
    stopFlag = False
    x = torch.arange(start=1, end=401).unsqueeze(0).unsqueeze(0).to("cuda:0")
    start = time.time()
    for epoch in range(num):
        adjust_learning_rate(optimizer, epoch, start_lr)
        for iter_num in range(num2):
            optimizer.zero_grad()
            output_tensor = myNet(Y)
            temp1 = output_tensor[:1, :1, 0:1]
            temp2 = output_tensor[:1, :1, 1:2]
            temp3 = output_tensor[:1, :1, 2:3]
            temp4 = output_tensor[:1, :1, 3:4]

            e1 = temp1
            e2 = temp2
            e3 = temp3
            e4 = temp4

            I_generate = (
                    2 - 4 * N * e1 - 2 * (N - 2 * e1) * cos_delta_2
                    + (C * (1 - 2 * e3) + 2 * e2) * cos_delta_1Minus2
                    - (C * (1 + 2 * e3) - 2 * e2) * cos_delta_1Plus2
                    + (S * (1 - 2 * e3)) * sin_delta_1Minus2
                    - (S * (1 + 2 * e3)) * sin_delta_1Plus2
                    + 4 * cos_delta_1 * (-1 * N * e2 + C * e4)
                    + 4 * sin_delta_1 * S * e4)
            error = (Y - I_generate)

            loss = torch.norm(error[:1, :1, int(range1 * sample_points):int(range2 * sample_points)], p=2, dim=2).squeeze(0)

            loss.backward()
            optimizer.step()
            running_loss[epoch * num2 + iter_num] = loss.item()
            if epoch * num2 + iter_num > 20:
                if stopCriterion(running_loss, loss.item(), epoch, num2, iter_num, t=0.0002):
                    stopFlag = True
                    break
        if stopFlag:
            # print('11111')
            break
            # print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')

    print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')
    end = time.time()
    time_cost = end - start
    print('time cost: ', time_cost, 's \n')

    return e1, e2, e3, e4, I_generate, running_loss, time_cost


def outNCS(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
           cos_delta_1, sin_delta_1,
           e1, e2, e3, e4,
           layers=4, hidden=2000, num=2, num2=300, f_set=0.05, beta=0.01, gama=1, range1=0.2, range2=0.8, seed = 5):
    """使用不考虑误差项的物理模型
    :param M_basis:
    """
    # 定义网络、损失函数、优化器
    series_num = int(M_basis.shape[2] * f_set)
    zero_col = torch.zeros([1, 1, M_basis.shape[2] - series_num]).to('cuda:0')
    sample_points = int(Y.shape[2])
    setup_seed(seed)
    myNet = linear_net(sample_points, 3 * series_num, layers, hidden).to("cuda:0")
    # myNet = ResNet(3 * series_num).to("cuda:0")
    # myNet = RNN(1, 400, 2, 3 * series_num).to("cuda:0")
    ### 记得改存储路径！！！！
    start_lr = 0.01
    optimizer = torch.optim.Adam(myNet.parameters(), lr=start_lr)
    # 开始训练
    running_loss = np.zeros(num * num2)
    stopFlag = False
    x = torch.arange(start=1, end=401).unsqueeze(0).unsqueeze(0).to("cuda:0")
    start = time.time()
    for epoch in range(num):
        adjust_learning_rate(optimizer, epoch, start_lr)
        for iter_num in range(num2):
            optimizer.zero_grad()
            output_tensor = myNet(Y)
            temp1 = torch.cat([output_tensor[:1, :1, 0:series_num], zero_col], dim=2)
            temp2 = torch.cat([output_tensor[:1, :1, series_num:2 * series_num], zero_col], dim=2)
            temp3 = torch.cat([output_tensor[:1, :1, 2 * series_num:3 * series_num], zero_col], dim=2)

            N_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp1])
            C_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp2])
            S_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp3])

            I_generate = (
                    2 - 4 * N_generate * e1 - 2 * (N_generate - 2 * e1) * cos_delta_2
                    + (C_generate * (1 - 2 * e3) + 2 * e2) * cos_delta_1Minus2
                    - (C_generate * (1 + 2 * e3) - 2 * e2) * cos_delta_1Plus2
                    + (S_generate * (1 - 2 * e3)) * sin_delta_1Minus2
                    - (S_generate * (1 + 2 * e3)) * sin_delta_1Plus2
                    + 4 * cos_delta_1 * (-1 * N_generate * e2 + C_generate * e4)
                    + 4 * sin_delta_1 * S_generate * e4)
            error = (Y - I_generate)

            loss = (torch.norm(error[:1, :1, int(range1 * sample_points):int(range2 * sample_points)], p=2, dim=2).squeeze(0)
                    + beta * torch.norm(output_tensor, p=1, dim=2).squeeze(0)
                    + gama * torch.norm(N_generate ** 2 + C_generate ** 2 +
                                        S_generate ** 2 - 1, p=2, dim=2).squeeze(0))

            loss.backward()
            optimizer.step()
            running_loss[epoch * num2 + iter_num] = loss.item()
            if epoch * num2 + iter_num > 20:
                if stopCriterion(running_loss, loss.item(), epoch, num2, iter_num, t=0.0002):
                    stopFlag = True
                    break
        if stopFlag:
            # print('11111')
            break
            # print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')

    print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')
    end = time.time()
    time_cost = end - start
    print('time cost: ', time_cost, 's \n')

    return N_generate, C_generate, S_generate, I_generate, running_loss, time_cost


def DIPSP(M_basis, Y, cos_delta_2, cos_delta_1Minus2, cos_delta_1Plus2, sin_delta_1Minus2, sin_delta_1Plus2,
              num=2, num2=300, f_set=0.05, beta=0.01, range1=0.2, range2=0.8, seed=5):
    """使用不考虑误差项的物理模型
    :param M_basis:
    """
    # 定义网络、损失函数、优化器
    series_num = int(M_basis.shape[2] * f_set)
    zero_col = torch.zeros([1, 1, M_basis.shape[2] - series_num]).to('cuda:0')
    sample_points = int(Y.shape[2])
    setup_seed(seed)
    myNet = ResNet(3 * series_num).to("cuda:0")
    criterion = nn.MSELoss().to("cuda:0")
    start_lr = 0.01
    optimizer = torch.optim.Adam(myNet.parameters(), lr=start_lr)
    # 开始训练
    running_loss = np.zeros(num * num2)
    start = time.time()
    for epoch in range(num):
        adjust_learning_rate(optimizer, epoch, start_lr)
        stopFlag = False
        for iter_num in range(num2):
            optimizer.zero_grad()
            output_tensor = myNet(Y)
            temp1 = torch.cat([output_tensor[:1, :1, 0:series_num], zero_col], dim=2)
            temp2 = torch.cat([output_tensor[:1, :1, series_num:2 * series_num], zero_col], dim=2)
            temp3 = torch.cat([output_tensor[:1, :1, 2 * series_num:3 * series_num], zero_col], dim=2)

            N_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp1])
            C_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp2])
            S_generate = torch.einsum('ijk,ilk->ilj', [M_basis, temp3])

            I_generate = 2 - 2 * N_generate * cos_delta_2 + C_generate * cos_delta_1Minus2 \
                         - C_generate * cos_delta_1Plus2 \
                         + S_generate * sin_delta_1Minus2 - S_generate * sin_delta_1Plus2
            error = (Y - I_generate)

            loss = (torch.norm(error[:1, :1, int(range1 * sample_points):int(range2 * sample_points)], p=2, dim=2).squeeze(0)
                    + beta * torch.norm(output_tensor, p=1, dim=2).squeeze(0))

            loss.backward()
            optimizer.step()
            running_loss[epoch * num2 + iter_num] = loss.item()
            if epoch * num2 + iter_num > 200:
                if stopCriterion(running_loss, loss.item(), epoch, num2, iter_num, t=0.00001):
                    stopFlag = True
                    break
        if stopFlag:
            # print('11111')
            break
        #     if epoch * num2 + iter_num > 0:
        #         out_N_simple = N_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         out_C_simple = C_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         out_S_simple = S_generate.cpu().squeeze(0).squeeze(0).detach().numpy()
        #         x = pd.DataFrame({'out_N_simple': out_N_simple, 'out_C_simple': out_C_simple,
        #                           'out_S_simple': out_S_simple})
        #         x.to_csv(r"D:\desktop\ysl_file\NCS_Measurement\finalPythonProject\net1data-0.1\NCSsimple-"+
        #                  str(epoch * num2 + iter_num) + "-" + str(loss.item()) + ".csv")
            # print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')

    print('iter num: %d, loss=%f' % (epoch * num2 + iter_num + 1, loss.item()), '\n')
    end = time.time()
    time_cost = end - start
    print('time cost: ', time_cost, 's \n')

    return N_generate, C_generate, S_generate, I_generate, running_loss, time_cost