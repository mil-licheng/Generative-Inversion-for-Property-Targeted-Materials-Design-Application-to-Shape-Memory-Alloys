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
import random


# define dataset class for data loader
class PFDataset(Dataset):
    """
    图片直接是256*256的3列矩阵，近似按灰度读取
    标签读取为长度为42的一维数组
    """
    # 设置加载的data和label,以及数据预处理方法
    def __init__(self, data1, transform=None):
        """
        Args:
            data1 (dataframe): 存储了微观图像地址和相应的应力应变曲线表格
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.landmarks_frame = data1
        self.transform = transform
    # 获取数据集大小
    def __len__(self):
        return len(self.landmarks_frame)
    # 根据索引获取一条训练的数据和标签
    def __getitem__(self, idx):
        #get rid of this
        if torch.is_tensor(idx):
            idx = idx.tolist()
        fea1 = self.landmarks_frame.iloc[idx,0]
        fea1 = np.array(fea1)
        fea1 = fea1.astype(np.float32)
        fea2 = self.landmarks_frame.iloc[idx,1]
        fea2 = np.array(fea2)
        fea2 = fea2.astype(np.float32)
        fea3 = self.landmarks_frame.iloc[idx,2]
        fea3 = np.array(fea3)
        fea3 = fea3.astype(np.float32)
        properties = self.landmarks_frame.iloc[idx,3]
        properties = np.array(properties)
        properties = properties.astype(np.float32)
        sample = {'fea1': fea1,'fea2': fea2,'fea3': fea3, 'properties': properties}
        if self.transform:
            sample = self.transform(sample)

        return sample


class MyModel(nn.Module):
    def __init__(self,out1,out2,hidden_size,trans_dimension,layer_numbers=5):
        super(MyModel, self).__init__()
        out1,out2 = out1,out2
        trans_dimension = trans_dimension
        # 第1.1部分网络
        self.fc1_1 = nn.Linear(10, trans_dimension)
        self.relu1_1 = nn.ReLU()
        self.fc2_1 = nn.Linear(trans_dimension, out1)
        self.relu2_1 = nn.ReLU()

        # 第1.2部分网络
        self.fc1_2 = nn.Linear(10, trans_dimension)
        self.relu1_2 = nn.ReLU()
        self.fc2_2 = nn.Linear(trans_dimension, out2)
        self.relu2_2 = nn.ReLU()

        # 第1.3部分网络
        self.fc1_3 = nn.Linear(4, trans_dimension)
        self.relu1_3 = nn.ReLU()
        self.fc2_3 = nn.Linear(trans_dimension, out2)
        self.relu2_3 = nn.ReLU()

        # 第2部分网络
        model = []
        # model.append(nn.Sequential(
        #     nn.Linear(out1+out2+out2, hidden_size),
        #     nn.ReLU()
        # ))
        model.append(nn.Sequential(
            nn.Linear(out1+out2+out2, hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU()
        ))
        for i in range(layer_numbers):
            model.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU()
            ))
            hidden_size = hidden_size//2

        model.append(nn.Sequential(
            nn.Linear(hidden_size, 2),
        ))
        self.model = nn.Sequential(*model)

    def forward(self, x1, x2, x3):
        # 第一部分
        x1 = self.fc1_1(x1)
        x1 = self.relu1_1(x1)
        x1 = self.fc2_1(x1)
        x1 = self.relu2_1(x1)

        # 第二部分
        x2 = self.fc1_2(x2)
        x2 = self.relu1_2(x2)
        x2 = self.fc2_2(x2)
        x2 = self.relu2_2(x2)

        # 第二部分
        x3 = self.fc1_3(x3)
        x3 = self.relu1_3(x3)
        x3 = self.fc2_3(x3)
        x3 = self.relu2_3(x3)

        # 合并三部分的输出
        x = torch.cat((x1, x2, x3), dim=1)

        # 第三部分
        x = self.model(x)

        return x

class MyModel2(nn.Module):
    def __init__(self,input_size,hidden_size,layer_numbers=5):
        super(MyModel2, self).__init__()

        # 第2部分网络
        model = []
        # model.append(nn.Sequential(
        #     nn.Linear(out1+out2+out2, hidden_size),
        #     nn.ReLU()
        # ))
        model.append(nn.Sequential(
            nn.Linear(input_size, hidden_size//8),
            nn.ReLU(),
            nn.Linear(hidden_size//8, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size//4, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU()
        ))
        for i in range(layer_numbers):
            model.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU()
            ))
            hidden_size = hidden_size//2

        model.append(nn.Sequential(
            nn.Linear(hidden_size, 2),
        ))
        self.model = nn.Sequential(*model)

    def forward(self, x1, x2, x3):
        # 合并三部分的输出
        x = torch.cat((x1, x2, x3), dim=1)

        # 第三部分
        x = self.model(x)

        return x

class MyModel3(nn.Module):
    def __init__(self,input_size1,input_size2,hidden_size,layer_numbers0=4,layer_numbers=5,sub_number = 5):
        sub_number = sub_number
        super(MyModel3, self).__init__()
        # 定义第一个子网络，用于处理 fea1 和 fea2
        subnet1 = []
        hidden_size_start = hidden_size

        layer_config = {
            4: [hidden_size // 8, hidden_size // 4, hidden_size // 2, hidden_size],
            3: [hidden_size // 4, hidden_size // 2, hidden_size],
            2: [hidden_size // 2, hidden_size],
            1: [hidden_size]
        }

        # 根据 layer_numbers 的值来选择对应的层配置
        layer_sizes = layer_config.get(layer_numbers0, [])
        layers = []

        # 动态添加层
        input_size = input_size1
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = size

        # 创建子网络并添加到 subnet1
        subnet1.append(nn.Sequential(*layers))

        for i in range(layer_numbers):
            subnet1.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.LeakyReLU(0.2),
            ))
            hidden_size = hidden_size//2

        subnet1.append(nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(hidden_size, sub_number),
        ))
        self.subnet1 = nn.Sequential(*subnet1)

        # 定义第二个子网络，用于处理 z 和 fea3
        subnet2 = []
        hidden_size = hidden_size_start

        # 动态添加层
        layers = []
        input_size = sub_number+input_size2
        for size in layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.LeakyReLU(0.2))
            input_size = size

        # 创建子网络并添加到 subnet1
        subnet2.append(nn.Sequential(*layers))

        for i in range(layer_numbers):
            subnet2.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.LeakyReLU(0.2),
            ))
            hidden_size = hidden_size//2

        subnet2.append(nn.Sequential(
            # nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        ))
        self.subnet2 = nn.Sequential(*subnet2)

    def forward(self,fea):
        fea1 = fea[:, :10]
        fea2 = fea[:, 10:18]
        fea3 = fea[:, 18:]
        # 通过第一个子网络得到 z
        combined1 = torch.cat((fea1, fea3), dim=1)
        z = self.subnet1(combined1)

        # 通过第二个子网络得到 m
        combined2 = torch.cat((z, fea2), dim=1)
        m = self.subnet2(combined2)

        return z, m


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, prediction, target):
        mse_loss = nn.MSELoss()

        loss_col1 = mse_loss(prediction[:, 0], target[:, 0])
        loss_col2 = mse_loss(prediction[:, 1], target[:, 1])

        #total_loss = loss_col1 + 5 * loss_col2
        # total_loss = loss_col1 + 30 * loss_col2
        total_loss = loss_col1 + 100 * loss_col2

        return total_loss


class CustomLoss2(nn.Module):
    # 在损失中引入物理约束，例如应力增大时，相变温度应该增大
    def __init__(self):
        super(CustomLoss2, self).__init__()

    def forward(self, inputs,outputs,model):
        total_loss = 0
        # 计算单调性约束损失
        # 输入增加一个小量delta
        all_list = [0,3,5,6,20,1,2,4,9,7]
        increase_list = [3,5,6,7]
        decrease_list = [2,4,9]
        delta = random.randint(1, 5)
        for i in all_list:
            inputs_increased = inputs.clone()
            inputs_increased[:, i] += delta
            # # 原子总数回到100
            # inputs_increased[:,:10] = inputs_increased[:,:10]/inputs_increased[:,:10].sum(dim=1, keepdim=True) * 100

            if i in increase_list:
                # 相变温度升高的元素
                inputs_increased[:, 0] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0
                monotonic_loss = torch.sum(torch.relu(outputs[:, 0] - outputs_increased[:, 0]))
            elif i in decrease_list:
                # 相变温度降低的元素
                inputs_increased[:, 1] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs大于outputs_increased损失为0
                monotonic_loss = torch.sum(torch.relu(outputs_increased[:, 0] - outputs[:, 0]))
            elif i == 20:
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu(outputs - outputs_increased))
            elif i == 0:
                # Ti增加，Ni减少
                inputs_increased[:, 1] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu(outputs[:, 0] - outputs_increased[:, 0]))
            elif i == 1:
                # Ni增加，Ti减少
                inputs_increased[:, 0] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu(outputs_increased[:, 0] - outputs[:, 0]))
            # 总损失
            total_loss = total_loss + monotonic_loss
            # 减少约束
        return total_loss

class CustomLoss3(nn.Module):
    # 在损失中引入物理约束，例如应力增大时，相变温度应该增大
    def __init__(self):
        super(CustomLoss3, self).__init__()

    def forward(self, inputs,outputs,model):
        total_loss = 0
        # 计算单调性约束损失
        # 输入增加一个小量delta
        all_list = [0,3,5,6,20,1,2,4,9,7]
        increase_list = [3,5,6,7]
        decrease_list = [2,4]
        const_list = [18.4,-109.3,-5,20,-23,10.9,5.3,7.6,0,0]
        delta = random.randint(1, 5)
        for i in all_list:
            inputs_increased = inputs.clone()
            inputs_increased[:, i] += delta
            # # 原子总数回到100
            # inputs_increased[:,:10] = inputs_increased[:,:10]/inputs_increased[:,:10].sum(dim=1, keepdim=True) * 100

            if i in increase_list:
                # 相变温度升高的元素
                inputs_increased[:, 0] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0
                monotonic_loss = torch.sum(torch.relu((outputs[:, 0]+delta*const_list[i]) - outputs_increased[:, 0]))
            elif i in decrease_list:
                # 相变温度降低的元素
                inputs_increased[:, 1] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs大于outputs_increased损失为0
                monotonic_loss = torch.sum(torch.relu(outputs_increased[:, 0] - (outputs[:, 0]+delta*const_list[i])))
            elif i == 20:
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu(outputs - outputs_increased))
            elif i == 0:
                # Ti增加，Ni减少
                inputs_increased[:, 1] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu((outputs[:, 0]+delta*const_list[i]) - outputs_increased[:, 0]))
            elif i == 1:
                # Ni增加，Ti减少
                inputs_increased[:, 0] -= delta
                _, outputs_increased = model(inputs_increased)
                # 计算单调性损失，outputs小于outputs_increased损失为0,应力增加，相变温度和输出功均增加
                monotonic_loss = torch.sum(torch.relu(outputs_increased[:, 0] - (outputs[:, 0]+delta*const_list[i])))
            # 总损失
            total_loss = total_loss + monotonic_loss
            # 减少约束
        return total_loss


def evalute(model,test_data,criterion,device):
    fea1 = torch.tensor(test_data.iloc[:, 0].tolist())
    fea2 = torch.tensor(test_data.iloc[:, 1].tolist())
    fea3 = torch.tensor(test_data.iloc[:, 2].tolist())
    prop = torch.tensor(test_data.iloc[:, 3].tolist())

    fea1 = fea1.to(device)
    fea2 = fea2.to(device)
    fea3 = fea3.to(device)
    prop = prop.to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(fea1,fea2,fea3)  # 计算测试数据的输出logits
        loss = criterion(outputs, prop).cpu().detach().numpy()

    outputs3 = outputs.cpu().detach().numpy()
    properties = prop.cpu().detach().numpy()

    # 最终真实值和预测值
    return properties,outputs3
def evalute3(model,test_data,criterion,device):
    fea1 = torch.tensor(test_data.iloc[:, 0].tolist())
    fea2 = torch.tensor(test_data.iloc[:, 1].tolist())
    fea3 = torch.tensor(test_data.iloc[:, 2].tolist())
    prop = torch.tensor(test_data.iloc[:, 3].tolist())

    fea = torch.cat([fea1, fea2, fea3], dim=1)
    # 维度转换
    fea = fea.to(device)
    prop = prop.to(device)

    model.eval()
    with torch.no_grad():
        _,outputs = model(fea)  # 计算测试数据的输出logits

    outputs3 = outputs.cpu().detach().numpy()
    properties = prop.cpu().detach().numpy()

    # 最终真实值和预测值
    return properties,outputs3

