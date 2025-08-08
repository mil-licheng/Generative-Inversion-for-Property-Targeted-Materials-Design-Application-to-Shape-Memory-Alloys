import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import optim
from tqdm import tqdm
from sklearn.metrics import r2_score
from ann_predict import PFDataset,MyModel,CustomLoss,evalute,MyModel2,CustomLoss2,MyModel3,CustomLoss3
from tools import data_processing,PFDataset
import argparse
import random

"""
MyModel3，模型架构，x的前10维fea1和最后4维fea2输入网络得到8维向量z，z和x的11到20维fea3组合起来，
经过网络得到2维向量m。损失函数loss包括两部分，loss1是z的前两维和y的mse，loss2是m和y的mse，loss=loss1+loss2
使用早停法，保存模型的最佳状态
"""



if __name__ == '__main__':
    # 设置命令行输入
    parser = argparse.ArgumentParser()
    parser.description = 'please enter parameters：num_epochs'
    parser.add_argument("-hidden_size", "--hidden_size", help="this is parameter hidden_size", dest="hidden_size", type=int
                        , default="512")
    parser.add_argument("-layer_numbers0", "--layer_numbers0", help="this is parameter layer_numbers0", dest="layer_numbers0", type=int
                        , default="1")
    parser.add_argument("-seed", "--seed", help="this is parameter seed", dest="seed", type=int
                        , default="50")
    args = parser.parse_args()

    # 读取文件，数据预处理
    path = '/share/home/3121102080/SME/'
    # path = 'H:/licheng/ML_HTSMAs/'
    ori_path = path + 'SME3.csv'
    ori_data = pd.read_csv(filepath_or_buffer=ori_path)
    ori_data = ori_data.fillna(0)

    print(ori_data.shape[0])
    # 加入反馈迭代的成分
    iter = 1
    for i in range(1,iter):
        fea_prop_iter_data = pd.read_csv(filepath_or_buffer=path+'iter{}.csv'.format(i))
        fea_prop_iter_data = fea_prop_iter_data.fillna(0)
        ori_data = pd.concat([ori_data,fea_prop_iter_data])
    # 创建一个布尔索引，找到 'Ms/℃' 列中值大于 300 的行
    mask = ori_data['Ms/℃'] > 280
    # mask = (ori_data['Hf']+ori_data['Zr']) > 18
    # 复制这些行三次
    new_rows = ori_data[mask].copy()
    ori_data = pd.concat([ori_data] + [new_rows] * 3, ignore_index=True)
    ori_data = ori_data.reset_index(drop=True)
    print(ori_data.shape[0])

    fea_prop_data = data_processing(ori_data)
    # fea_prop_data.to_pickle('fea_prop_data.pkl')
    #设置随机数种子
    seed = 195
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    batch_size = 32
    # device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    learn_rate = 0.002
    num_epochs = 170

    hidden_size = args.hidden_size
    layer_numbers = 2
    layer_numbers0 = args.layer_numbers0

    save_path = path + "ann_{}_{}_{}_3_{}.pth".format(iter+1,hidden_size,layer_numbers,layer_numbers0) # 模型权重参数保存位置

    # 创建dataset，加载dataloader
    train_set = PFDataset(fea_prop_data)
    trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=1, pin_memory=True)

    # model = MyModel(inum,inum,3072,layer_numbers=layer_numbers,trans_dimension=trans_dimension)
    # model = MyModel2(input_size=22,hidden_size=hidden_size, layer_numbers=layer_numbers)
    # model = MyModel3(input_size1=14,input_size2=8,hidden_size=hidden_size, layer_numbers=layer_numbers,sub_number=sub_number)
    model = MyModel3(input_size1=14, input_size2=8, hidden_size=hidden_size, layer_numbers=layer_numbers,
                     layer_numbers0=layer_numbers0,sub_number=8)
    model.to(device)
    # 定义损失函数
    # loss_func = nn.MSELoss()
    loss_func = CustomLoss()
    loss_func2 = CustomLoss2()
    # loss_func2 = CustomLoss3()
    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)
    # 模型训练
    best_acc, best_acc_v, best_epoch = 0.0, 0.0, 0  # 最好准确度，出现的轮数
    global_step = 0  # 全局的step步数，用于画图
    train_loss, train_R2, test_loss, test_R2 = [], [], [], []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        epoch_train_loss, epoch_train_R2 = 0.0, 0.0  # 总损失值初始化为0
        epoch_predictions, epoch_targets = [], []

        # 循环读取训练数据集，更新模型参数，sample读取数据和自定义的打包方式相关
        for step, sample in enumerate(trainloader):
            # 将图片和标签送入计算设备
            fea1 = sample['fea1']
            fea2 = sample['fea2']
            fea3 = sample['fea3']
            properties = sample['properties']
            fea = torch.cat([fea1, fea2, fea3], dim=1)
            # 维度转换
            fea = fea.to(device)

            # curves = torch.stack(curves, dim=1)
            properties = properties.to(device)
            # 梯度初始化为0
            optimizer.zero_grad()
            # 训练后的输出
            sub_outputs,outputs = model(fea)
            # 计算损失
            loss = 0.8*loss_func(outputs, properties) + 0.2*loss_func(sub_outputs[:, :2], properties)
            # 计算单调性损失
            loss = loss+5*loss_func2(fea,outputs,model)
            # 输出R2
            outputs3 = outputs.cpu().detach().numpy()
            properties = properties.cpu().detach().numpy()
            # R2_train = r2_score(outputs3, properties)
            # epoch_train_R2 += R2_train
            # 拼接每个batch的结果
            epoch_predictions.append(outputs3)
            epoch_targets.append(properties)
            # print(r2_score(outputs3, curves3))
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
            # 累计训练损失
            epoch_train_loss += loss.item()

        train_loss.append(epoch_train_loss / len(trainloader))
        # train_R2.append(epoch_train_R2 / len(trainloader))
        epoch_predictions = np.concatenate(epoch_predictions)
        epoch_targets = np.concatenate(epoch_targets)
        r2 = r2_score(epoch_targets, epoch_predictions)
        r2_1 = r2_score(epoch_targets, epoch_predictions,multioutput='raw_values')
        train_R2.append(r2)

        if train_R2[-1] > best_acc:
            # 保存最好数据
            best_acc = train_R2[-1]
            # 保存最好的模型参数值状态
            # torch.save(model.state_dict(), save_path)
            best_epoch = epoch

            torch.save(model.state_dict(), save_path)

    data_results = pd.DataFrame({'loss':train_loss,'R2':train_R2,'v_R2':test_R2})
    data_results.to_pickle(path + 'data_results{}_{}_{}.pkl'.format(iter+1,hidden_size,layer_numbers))
