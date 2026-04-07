import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from functools import reduce
from functools import partial

from .basics import SpectralConv1d, SpectralConv1d_filter
import numpy as np
from scipy.signal import butter, filtfilt


class FNN1d(nn.Module):
    def __init__(self, modes, width, layers=None):
        super(FNN1d, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(2, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc1 = nn.Linear(layers[-1], 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.relu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class FNN1d_DOF(nn.Module):
    def __init__(self, modes, width, layers=None):
        super(FNN1d_DOF, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        if layers is None:
            layers = [width] * 2

        self.fc0 = nn.Linear(5, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc1 = nn.Linear(layers[-1], 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        length = len(self.ws)
        print(x.shape)
        os.system('pause')

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        return x


class FNN1d_VTCD(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_VTCD, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)
        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_VTCD_GradNorm(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_VTCD_GradNorm, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)
        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x

    def get_last_layer(self):
        return self.fc3

class FNN1d_FES(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_FES, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_BSA(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, layers=None):
        super(FNN1d_BSA, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x


class FNN1d_BSA_GradNorm(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_BSA_GradNorm, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 3

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        return x

    def get_last_layer(self):
        return self.fc3

class FNN1d_ANN(nn.Module):
    def __init__(self, inputDim, outputDim, layers=None):
        super(FNN1d_ANN, self).__init__()

        """
        Simple fully connected networks. It contains several layers of ordinary neural layers.
        
        """

        self.inputDim = inputDim
        self.outputDim = outputDim

        self.fc_width = 48
        self.fc0 = nn.Linear(self.inputDim, self.fc_width)
        self.fc1 = nn.Linear(self.fc_width, self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.fc_width)
        self.fc4 = nn.Linear(self.fc_width, self.outputDim)
        # self.bn1 = nn.BatchNorm1d(2500)
        # self.bnW = nn.BatchNorm1d(width)

    def forward(self, x):
        x = self.fc0(x)
        x = F.elu(x)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc4(x)
        return x


class FNN1d_HM(nn.Module):
    def __init__(self, modes, width, fc_dim, inputDim, outputDim, task_number, layers=None):
        super(FNN1d_HM, self).__init__()

        """
        The overall network. It contains several layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.inputDim = inputDim
        self.outputDim = outputDim

        if layers is None:
            layers = [width] * 4

        self.fc0 = nn.Linear(self.inputDim, layers[0])  # input channel is 2: (a(x), x)

        self.sp_convs = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers, layers[1:])])

        self.ws = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers, layers[1:])])

        self.fc_width = fc_dim
        self.fc1 = nn.Linear(layers[-1], self.fc_width)
        self.fc2 = nn.Linear(self.fc_width, self.fc_width)
        self.fc3 = nn.Linear(self.fc_width, self.fc_width)
        self.fc4 = nn.Linear(self.fc_width, self.outputDim)

        self.bn1 = nn.BatchNorm1d(2500)
        self.bnW = nn.BatchNorm1d(width)
        self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())

    def forward(self, x):
        length = len(self.ws)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs, self.ws)):
            x1 = speconv(x)
            x2 = w(x)
            x = x1 + x2
            if i != length - 1:
                x = F.elu(x)

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        x = self.fc3(x)
        x = F.elu(x)
        x = self.fc4(x)
        return x

    def get_last_layer(self):
        return self.fc4


class CNN_GRU(nn.Module):
    def __init__(self, channel, input_dim, output_dim, kernel_size):
        super(CNN_GRU, self).__init__()
        self.channel = channel
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.padding = int(kernel_size / 2 - 1)
        self.CNN1 = nn.Conv1d(in_channels=input_dim, out_channels=20, kernel_size=self.kernel_size,
                              stride=1,
                              padding=50)
        self.CNN2 = nn.Conv1d(in_channels=20, out_channels=40, kernel_size=self.kernel_size,
                              stride=1,
                              padding=49)
        self.CNN3 = nn.Conv1d(in_channels=40, out_channels=50, kernel_size=self.kernel_size,
                              stride=1,
                              padding=50)
        self.CNN4 = nn.Conv1d(in_channels=50, out_channels=50, kernel_size=self.kernel_size,
                              stride=1,
                              padding=49)
        # self.GRU1 = nn.GRU(input_size=50, hidden_size=50, num_layers=5, batch_first=True, bidirectional=False)
        # self.GRU2 = nn.GRU(input_size=50, hidden_size=43, num_layers=5, batch_first=True, bidirectional=False)
        self.GRU1 = nn.GRU(input_size=self.input_dim, hidden_size=30, num_layers=2, batch_first=True,
                           bidirectional=False)
        self.GRU2 = nn.GRU(input_size=30, hidden_size=50, num_layers=2, batch_first=True,
                           bidirectional=False)
        self.GRU3 = nn.GRU(input_size=50, hidden_size=self.output_dim, num_layers=2, batch_first=True,
                           bidirectional=False)
        # self.CNN1 = self.init_CNN(self.CNN1)
        # self.CNN2 = self.init_CNN(self.CNN2)
        # self.CNN3 = self.init_CNN(self.CNN3)
        # self.CNN4 = self.init_CNN(self.CNN4)
        self.GRU1 = self.init_GRU(self.GRU1)
        self.GRU2 = self.init_GRU(self.GRU2)
        self.GRU3 = self.init_GRU(self.GRU3)

    def forward(self, x):
        # x = x.permute(0, 2, 1)
        # x = self.CNN1(x)
        # x = self.CNN2(x)
        # x = self.CNN3(x)
        # x = self.CNN4(x)
        # x = x.permute(0, 2, 1)
        x, _ = self.GRU1(x)
        x, _ = self.GRU2(x)
        x, _ = self.GRU3(x)
        x = x.permute(0, 2, 1)
        return x

    def init_GRU(self, net):
        for name, p in net.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.constant_(p, 0)
        return net

    def init_CNN(self, net):
        nn.init.kaiming_uniform_(net.weight)
        return net


# 自定义低通滤波器类
class LowPassFilter(nn.Module):
    def __init__(self, fs, cutoff, order):
        super(LowPassFilter, self).__init__()
        
        # 计算正规化截止频率
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist

        # 使用Butterworth滤波器设计滤波器系数
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        # 转换为张量
        self.b = torch.tensor(b, dtype=torch.float32)
        self.a = torch.tensor(a, dtype=torch.float32)
        
    def forward(self, x):
        # 将输入转换为张量
        # x = torch.tensor(x, dtype=torch.float32)
        
        # 将滤波器系数复制到与输入张量相同的设备上
        device = x.device
        b = self.b.to(device)
        a = self.a.to(device)
        # x = x.cpu().numpy()        
        # x_copy = x.copy()
        # 将滤波器应用于输入信号
        filtered_x = filtfilt(b.detach().cpu().numpy(), a.detach().cpu().numpy(), x.detach().cpu().numpy())
        filtered_x = filtered_x.copy()
        # 将滤波后的信号转换回张量并返回
        return torch.tensor(filtered_x, dtype=torch.float32, device=device)



class FNN1d_VTCD_GradNorm_Branch(nn.Module):
    def __init__(self, modes1, modes2, width1, width2, fc_dim1, fc_dim2, inputDim1, inputDim2, outputDim1, outputDim2, task_number, layers1=None, layers2=None):
        super(FNN1d_VTCD_GradNorm_Branch, self).__init__()

        """
        The overall network with two branches. Each branch contains several layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output1: the solution of the first output task
        output shape1: (batchsize, x=s, c=outputDim1)
        output2: the solution of the second output task
        output shape2: (batchsize, x=s, c=outputDim2)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width1 = width1
        self.width2 = width2
        self.inputDim1 = inputDim1
        self.inputDim2 = inputDim2
        self.outputDim1 = outputDim1
        self.outputDim2 = outputDim2
        # 创建低通滤波器层
        self.lowpass = LowPassFilter(fs=1000, cutoff=40, order=4)

        if layers1 is None:
            layers1 = [width1] * 4
            
        if layers2 is None:
            layers2 = [width2] * 4

        self.fc0_1 = nn.Linear(self.inputDim1, layers1[0])  # input channel is 2: (a(x), x)
        self.fc0_2 = nn.Linear(self.inputDim2, layers2[0])  

        self.sp_convs1 = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes1) for in_size, out_size in zip(layers1, layers1[1:])])
        self.sp_convs2 = nn.ModuleList([SpectralConv1d(
            in_size, out_size, self.modes2) for in_size, out_size in zip(layers2, layers2[1:])])

        self.ws1 = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers1, layers1[1:])])
        self.ws2 = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                            for in_size, out_size in zip(layers2, layers2[1:])])

        self.fc_width1 = fc_dim1
        self.fc_width2 = fc_dim2
        
        self.fc1_1 = nn.Linear(layers1[-1], self.fc_width1)
        self.fc2_1 = nn.Linear(self.fc_width1, self.fc_width1)
        # self.fc3_1 = nn.Linear(self.fc_width1, self.fc_width1)
        # self.fc4_1 = nn.Linear(self.fc_width1, self.fc_width1)
        # self.fc5_1 = nn.Linear(self.fc_width1, self.outputDim1)
        self.fc3_1 = nn.Linear(self.fc_width1, self.outputDim1)
        
        self.fc1_2 = nn.Linear(layers2[-1], self.fc_width2)
        self.fc2_2 = nn.Linear(self.fc_width2, self.fc_width2)
        # self.fc3_2 = nn.Linear(self.fc_width2, self.fc_width2)
        # self.fc4_2 = nn.Linear(self.fc_width2, self.outputDim2)
        self.fc3_2 = nn.Linear(self.fc_width2, self.outputDim2)
        
        # self.bn1 = nn.BatchNorm1d(2500)
        # self.bnW = nn.BatchNorm1d(width1)
        # self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())


    def forward(self, x):
        length1 = len(self.ws1)

        x_1 = self.fc0_1(x)
        x_1 = x_1.permute(0, 2, 1)
        x_1 = self.lowpass(x_1)        # 先不用低通滤波试一下呢

        for i, (speconv, w) in enumerate(zip(self.sp_convs1, self.ws1)):
            x1 = speconv(x_1)
            x2 = w(x_1)
            x_1 = x1 + x2
            if i != length1 - 1:
                x_1 = F.elu(x_1)

        x_1 = x_1.permute(0, 2, 1)
        x_1 = self.fc1_1(x_1)
        x_1 = F.elu(x_1)
        x_1 = self.fc2_1(x_1)
        x_1 = F.elu(x_1)
        # x_1 = self.fc3_1(x_1)
        # x_1 = F.elu(x_1)
        # x_1 = self.fc4_1(x_1)  # First output branch
        # x_1 = F.elu(x_1)
        # output1 = self.fc5_1(x_1)  # First output branch
        
        output1 = self.fc3_1(x_1)  # First output branch
        
        # output1 = output1.permute(0, 2, 1)
        # output1 = self.lowpass(output1)
        # output1 = output1.permute(0, 2, 1)
        # device = torch.device("cuda")
        # output1 = output1.to(device)


        length2 = len(self.ws2)

        x_2 = self.fc0_2(x)
        x_2 = x_2.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs2, self.ws2)):
            x1 = speconv(x_2)
            x2 = w(x_2)
            x_2 = x1 + x2
            if i != length2 - 1:
                x_2 = F.elu(x_2)

        x_2 = x_2.permute(0, 2, 1)
        x_2 = self.fc1_2(x_2)
        x_2 = F.elu(x_2)
        x_2 = self.fc2_2(x_2)
        x_2 = F.elu(x_2)
        # x_2 = self.fc3_2(x_2)
        # x_2 = F.elu(x_2)
        # output2 = self.fc4_2(x_2)  # Second output branch
        output2 = self.fc3_2(x_2)  # Second output branch
          
        return output1, output2


class FNN1d_VTCD_GradNorm_Branch_filter(nn.Module):
    def __init__(self, modes1, modes2, width1, width2, fc_dim1, fc_dim2, inputDim1, inputDim2, outputDim1, outputDim2, lowpass_fc, task_number, layers1=None, layers2=None):
        super(FNN1d_VTCD_GradNorm_Branch_filter, self).__init__()

        """
        The overall network with two branches. Each branch contains several layers of the Fourier layer.
        1. Lift the input to the desired channel dimension by self.fc0.
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.
        3. Project from the channel space to the output space by self.fc1 and self.fc2.

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output1: the solution of the first output task
        output shape1: (batchsize, x=s, c=outputDim1)
        output2: the solution of the second output task
        output shape2: (batchsize, x=s, c=outputDim2)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width1 = width1
        self.width2 = width2
        self.inputDim1 = inputDim1
        self.inputDim2 = inputDim2
        self.outputDim1 = outputDim1
        self.outputDim2 = outputDim2
        self.lowpass_fc = lowpass_fc

        if layers1 is None:
            layers1 = [width1] * 4
            
        if layers2 is None:
            layers2 = [width2] * 4

        self.fc0_1 = nn.Linear(self.inputDim1, layers1[0])  # input channel is 2: (a(x), x)
        self.fc0_2 = nn.Linear(self.inputDim2, layers2[0])  # input channel is 2: (a(x), x)

        self.sp_convs1 = nn.ModuleList([SpectralConv1d_filter(
            in_size, out_size, self.modes1, self.lowpass_fc) for in_size, out_size in zip(layers1, layers1[1:])])
        self.sp_convs2 = nn.ModuleList([SpectralConv1d_filter(
            in_size, out_size, self.modes2, self.lowpass_fc) for in_size, out_size in zip(layers2, layers2[1:])])

        self.ws1 = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                                 for in_size, out_size in zip(layers1, layers1[1:])])
        self.ws2 = nn.ModuleList([nn.Conv1d(in_size, out_size, 1)
                            for in_size, out_size in zip(layers2, layers2[1:])])

        self.fc_width1 = fc_dim1
        self.fc_width2 = fc_dim2
        
        self.fc1_1 = nn.Linear(layers1[-1], self.fc_width1)
        self.fc2_1 = nn.Linear(self.fc_width1, self.fc_width1)
        self.fc3_1 = nn.Linear(self.fc_width1, self.outputDim1)
        
        self.fc1_2 = nn.Linear(layers2[-1], self.fc_width2)
        self.fc2_2 = nn.Linear(self.fc_width2, self.fc_width2)
        self.fc3_2 = nn.Linear(self.fc_width2, self.outputDim2)
        
        # self.bn1 = nn.BatchNorm1d(2500)
        # self.bnW = nn.BatchNorm1d(width1)
        # self.dropout = nn.Dropout(p=0.5)
        self.task_weights = torch.nn.Parameter(torch.ones(task_number).float())


    def forward(self, x):
        length1 = len(self.ws1)

        x_1 = self.fc0_1(x)
        x_1 = x_1.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs1, self.ws1)):
            x1 = speconv(x_1)
            x2 = w(x_1)
            x_1 = x1 + x2
            if i != length1 - 1:
                x_1 = F.elu(x_1)

        x_1 = x_1.permute(0, 2, 1)
        x_1 = self.fc1_1(x_1)
        x_1 = F.elu(x_1)
        x_1 = self.fc2_2(x_1)
        x_1 = F.elu(x_1)
        output1 = self.fc3_1(x_1)  # First output branch


        length2 = len(self.ws2)

        x_2 = self.fc0_2(x)
        x_2 = x_2.permute(0, 2, 1)

        for i, (speconv, w) in enumerate(zip(self.sp_convs2, self.ws2)):
            x1 = speconv(x_2)
            x2 = w(x_2)
            x_2 = x1 + x2
            if i != length2 - 1:
                x_2 = F.elu(x_2)

        x_2 = x_2.permute(0, 2, 1)
        x_2 = self.fc1_2(x_2)
        x_2 = F.elu(x_2)
        x_2 = self.fc2_2(x_2)
        x_2 = F.elu(x_2)
        output2 = self.fc3_2(x_2)  # Second output branch
        
        return output1, output2