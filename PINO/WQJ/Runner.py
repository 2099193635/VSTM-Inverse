from argparse import ArgumentParser
import yaml
import torch
from models.fourier1d import FNN1d_VTCD_GradNorm, FNN1d_VTCD, FNN1d_VTCD_GradNorm_Branch,FNN1d_VTCD_GradNorm_Branch_filter
from train_utils import Adam
from train_utils.datasets import VTCD_Loader_WithVirtualData
from train_utils.train_2d import train_VTCD_GradNorm,train_VTCD_Rail,train_VTCD_sep
# from train_utils.eval_2d import eval_burgers
# from train_utils.solution_extension import FDD_Extension
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from train_utils.losses_VTCD import derivatation
from train_utils.losses import LpLoss
# import spicy.io as io
import numpy as np
import time
from Defination_Experiments import Experiments_GradNorm_VTCD

f = open(r'configs/VTCD.yaml')
VTCD_config = yaml.safe_load(f)


# dataset = VTCD_Loader_WithVirtualData(datapath='F:\PINO_VTCD\data\\test_2v.mat', 
#                                           datapath_test='F:\PINO_VTCD\data\\test_2v.mat',
#                                           nt=2000, nSlice=500,
#                                           sub_t=1,
#                                           new=False, inputDim=135,
#                                           outputDim=35)
# print(dataset)

def run(config, args=False):
    data_config = config['data']
    ComDevice = torch.device('cuda:0')
    dataset = VTCD_Loader_WithVirtualData(data_config['datapath'],
                                          data_config['test_datapath'],
                                          nt=data_config['nt'], nSlice1=data_config['nSlice1'],nSlice2=data_config['nSlice2'],
                                          sub_t=data_config['sub_t'],
                                          new=False, inputDim1=data_config['inputDim1'],inputDim2=data_config['inputDim2'],
                                          outputDim1=data_config['outputDim1'],outputDim2=data_config['outputDim2'])
    
    train_loader, test_loader= dataset.make_loader(
        n_sample=data_config['n_sample'], 
        batch_size=config['train']['batchsize'], 
        start=data_config['offset'])
    if data_config['OperatorType'] == 'PINO-MBD' or data_config['OperatorType'] == 'PINO':
        if data_config['NoData'] == 'On':
            task_number = 1
        else:
            task_number = 2
            if data_config['DiffLossSwitch'] == 'On':
                task_number += 1
            if data_config['VirtualSwitch'] == 'On':
                task_number += 1


                
    else:
        task_number = 1

    print('This mission will have {} task(s)'.format(task_number))
    if data_config['GradNorm'] == 'On' and task_number != 1:
        print('GradNorm will be launched with alpha={}.'.format(data_config['GradNorm_alpha']))
    else:
        print('GradNorm will not be launched for this mission.')
    
    model = FNN1d_VTCD_GradNorm_Branch(modes1=config['model']['modes1'],modes2=config['model']['modes2'],
                        width1=config['model']['width1'], width2=config['model']['width2'],
                        fc_dim1=config['model']['fc_dim1'],fc_dim2=config['model']['fc_dim2'],
                        inputDim1=data_config['inputDim1'],inputDim2=data_config['inputDim2'],
                        outputDim1=data_config['outputDim1'],outputDim2=data_config['outputDim2'],
                        task_number=task_number).to(ComDevice)

    optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'])
    # optimizer = Adam(model.parameters(), betas=(0.9, 0.999), lr=config['train']['base_lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.75)
    
    train_VTCD_Rail(model,
                        train_loader, test_loader,
                        optimizer, scheduler,
                        config,
                        inputDim1=data_config['inputDim1'], inputDim2=data_config['inputDim2'], 
                        outputDim1=data_config['outputDim1'],outputDim2=data_config['outputDim2'],
                        D=data_config['D'], ComDevice=ComDevice,
                        rank=0, log=False,
                        project='PINO-VTCD',
                        group='default',
                        tags=['default'],
                        use_tqdm=True)
    
    
    return model

def test(config, eval_model, args=False):
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    data_config = config['data']
    dataset = VTCD_Loader_WithVirtualData(data_config['test_datapath'],
                                          data_config['test_datapath'],
                                          nt=data_config['nt'], nSlice=data_config['nSlice'],
                                          sub_t=data_config['sub_t'],
                                          new=False, inputDim=data_config['inputDim'],
                                          outputDim=data_config['outputDim'])
    # Manual:Change new to False(from new)
    test_loader = dataset.make_loader(n_sample=data_config['test_sample'],
                                                                 batch_size=config['train']['batchsize'],
                                                                 start=data_config['offset'])
    Index = 0
    # Define loss for all types of output
    Signal_Loss = 0.0
    First_Differential_Loss = 0.0
    Second_Differential_Loss = 0.0
    criterion = torch.nn.L1Loss(reduction='mean')
    for x, y in test_loader:
        device2 = torch.device('cpu')
        x, y = x.to(device2), y.to(device2)
        batch_size = config['train']['batchsize']
        inputDim = VTCD_data_config['inputDim']
        outputDim = VTCD_data_config['outputDim']
        nt = data_config['nt']
        out = model(x)
        Derivative_Data, Derivative_y  = derivatation(y, out, VTCD_data_config['D'])
        GroundTruth_Data = y.to(device2).permute([0, 2, 1])[0, :, :].detach().numpy()
        ModelOutput_Data = out.to(device2).permute([0, 2, 1])[0, :, :].detach().numpy()
        Signal_Loss += criterion(out, y[:, :, :35])
        First_Differential_Loss += criterion(Derivative_Data[:, :, :35], Derivative_y[:, :, :35])
        Second_Differential_Loss += criterion(Derivative_Data[:, :, 35:], Derivative_y[:, :, 35:])
        Index += 1
    Signal_Loss /= len(test_loader)
    First_Differential_Loss /= len(test_loader)
    Second_Differential_Loss /= len(test_loader)
    print('Signal_Loss:{};First_Differential_Loss:{};Second_Differential_Loss:{}'.format(Signal_Loss,
                                                                                         First_Differential_Loss,
                                                                                         Second_Differential_Loss))


# Style = 'Train'
Style = 'Train'
Multiple = 'Yes'
Clip = 1
File = './configs/VTCD.yaml'
if Style == 'Train':
    time.sleep(1)
    Experiments_GradNorm_VTCD(Multiple, Clip, File, run)
else:
    device = torch.device('cpu')
    VTCD_data_config = VTCD_config
    # model = FNN1d_VTCD_GradNorm(modes=VTCD_config['model']['modes'],
    #                    width=VTCD_config['model']['width'], fc_dim=VTCD_config['model']['fc_dim'],
    #                    inputDim=VTCD_data_config['inputDim'],
    #                    outputDim=VTCD_data_config['outputDim'], task_number=1).to(device)
    model = FNN1d_VTCD_GradNorm_Branch(modes1=VTCD_config['model']['modes1'],modes2=VTCD_config['model']['modes2'],
                        width1=VTCD_config['model']['width1'], width2=VTCD_config['model']['width2'],
                        fc_dim1=VTCD_config['model']['fc_dim1'],fc_dim2=VTCD_config['model']['fc_dim2'],
                        inputDim1=VTCD_config['data']['inputDim1'],inputDim2=VTCD_config['data']['inputDim2'],
                        outputDim1=VTCD_config['data']['outputDim1'],outputDim2=VTCD_config['data']['outputDim2'],
                        task_number=3).to(device)
    if 'ckpt' in VTCD_config['train']:
        ckpt_path = VTCD_config['train']['ckpt']
        ckpt = torch.load(ckpt_path)
        # o = 'F:/Pycharm/ExistingPytorch/GNN_Series/Physics-Informed Neural Operator/PINO-Project1/checkpoints/VTCDRunner/VTCD_FNO.pt'
        # print(ckpt)
        model.load_state_dict(ckpt['model'])
    data_config = VTCD_config['data']
    
    dataset = VTCD_Loader_WithVirtualData(data_config['datapath'],
                                          data_config['test_datapath'],
                                          nt=data_config['nt'], nSlice1=data_config['nSlice1'],nSlice2=data_config['nSlice2'],
                                          sub_t=data_config['sub_t'],
                                          new=False, inputDim1=data_config['inputDim1'],inputDim2=data_config['inputDim2'],
                                          outputDim1=data_config['outputDim1'],outputDim2=data_config['outputDim2'])
    
    train_loader, test_loader= dataset.make_loader(
        n_sample=data_config['n_sample'], 
        batch_size=data_config['batchsize'], 
        start=data_config['offset'])
    
    time_start = time.time() 
    Switch1 = 'On'
    if Switch1 == 'On':
        eval_iter = iter(test_loader)
        x, y = next(eval_iter)
        out = model(x)
        # print(out)
        # out = torch.tensor(out[1])
        out = torch.cat((torch.tensor(out[0]), torch.tensor(out[1])), dim=-1)
        # print(out1)
        SavePath = 'J:\PINO_Branch\checkpoints\VTCDRunner\Runner1\\'
        Name = 'PhysicsUninformed_Performance1.txt'
        Name = SavePath + Name
        output = torch.cat([out[1, :, :], torch.cat([y[1, :, :35], y[1, :, -24:]], dim=-1)],dim=-1)
        print(output.shape)
        np.savetxt(Name, output.numpy())
        
        
    Switch2 = 'Off'
    if Switch2 == 'On':
        # test(config=VTCD_config, eval_model=model)
        # Scale the losses for all different components
        Scale_loss = np.zeros((35, 3))
        batch_number = 0
        loss1_sum = 0
        loss2_sum = 0
        loss3_sum = 0
        myloss = LpLoss(size_average=True)
        for x, y in test_loader:
            print('Operating batch No.{}'.format(batch_number))
            out = model(x)
            out = torch.cat((torch.tensor(out[0]), torch.tensor(out[1])), dim=-1)
            # test_weights = PDE_weights_virtual.double().repeat_interleave(x.size(0), dim=0)
            # _, _, Derivative_Data = VTCD_PINO_loss(out, x, test_weights, ToOneV, VTCD_config['data']['inputDim'],
            #                                        VTCD_config['data']['inputDim'], VTCD_config['data']['D'], device)
            Derivative_Data, Derivative_y = derivatation(y, out, 4.5)
            De0_GT = y[:, :, :59]
            De1_GT = Derivative_y[:, :, :35]
            De2_GT = Derivative_y[:, :, 35:]
            De0_Pre = out[:, :, :59]
            De1_Pre = Derivative_Data[:, :, :35]
            De2_Pre = Derivative_Data[:, :, 35:]

            loss1 = myloss(De0_GT,De0_Pre) / len(test_loader)
            loss2 = myloss(De1_GT,De1_Pre) / len(test_loader)
            loss3 = myloss(De2_GT,De2_Pre)/ len(test_loader)
            for i in range(0, 35):
                for j in range(0, 3):
                    if j == 0:
                        Fruit1 = De0_GT[:, :, i]
                        Fruit2 = De0_Pre[:, :, i]
                    elif j == 1:
                        Fruit1 = De1_GT[:, :, i]
                        Fruit2 = De1_Pre[:, :, i]
                    elif j == 2:
                        Fruit1 = De2_GT[:, :, i]
                        Fruit2 = De2_Pre[:, :, i]
                    Scale_loss[i, j] = Scale_loss[i, j] + myloss(Fruit1, Fruit2).detach().numpy()

            batch_number += 1
        #     loss1_sum += loss1.item()
        #     loss2_sum += loss2.item()
        #     loss3_sum += loss3.item()
        
        # loss1_sum = loss1_sum / len(test_loader)
        # loss2_sum = loss2_sum / len(test_loader)
        # loss3_sum = loss3_sum / len(test_loader)
        # print(loss1_sum)
        # print(loss2_sum)
        # print(loss3_sum)
        Scale_loss = Scale_loss / batch_number
        print(Scale_loss)
        SavePath = 'J:\PINO_Branch\checkpoints\VTCDRunner\Runner1\\'
        Name = 'WithoutEN_Scale1.txt'
        Name = SavePath + Name
        np.savetxt(Name, Scale_loss)

# time_end = time.time()    
# time_c= time_end - time_start  
# print('time cost', time_c, 's')