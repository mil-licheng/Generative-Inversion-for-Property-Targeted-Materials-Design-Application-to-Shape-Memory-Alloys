import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from networks.c_p_WGAN_gp import Generator,Discriminator
from tools.function import transform_c_p
from tools.score import uniqueNew_score,get_score1,get_score2,get_score1_2
from networks.ann_predict import PFDataset,CustomLoss,evalute,MyModel,MyModel2,MyModel3
import time

"""
采样生成10000个样本, 并预测性能
"""


if __name__ == '__main__':
    # 设置随机数种子
    seed = 1008
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # 生成器和评估网络实例化
    # 定义超参数
    input_size_G,input_size_D = 10,19  # 输入噪声的维度
    hidden_size = 4096  # 隐藏层维度

    output_size_G,output_size_D = 19,1  # 输出维度
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    generator = Generator(input_size_G, hidden_size, output_size_G, 2).to(device)

    generator.load_state_dict(torch.load("networks/generator4096_94501_10_2_im_14_2.pth"))
    generator.eval()

    # 预测器
    hidden_size = 512
    iter = 1
    layer_numbers = 2

    save_path = "networks/ann_2_512_2_3_1.pth"
    model = MyModel3(input_size1=14, input_size2=8, hidden_size=512, layer_numbers=2,
                     layer_numbers0=1,sub_number=8)
    model.load_state_dict(torch.load(save_path))
    model.to(device)

    batch_size = 5*10**3
    ways = torch.tensor([[1, 0, 0]],device=device)
    ways = ways.expand(batch_size, -1)
    # 生成一些样本并可视化
    with torch.no_grad():
        generated_output = generator(torch.randn(batch_size, input_size_G, device=device))
        # 通过评估网络计算Ms和workout
        fea1 = generated_output[:, :10].clone()
        fea2 = generated_output[:, 10:18].clone()
        fea3 = generated_output[:, 18:].clone()

        fea1_T, fea2_T, fea3_T = transform_c_p(fea1, fea2, fea3)
        final_y = torch.cat([fea1_T, fea2_T, fea3_T], dim=1)
        fea3_T = torch.cat((fea3_T, ways), dim=1)
        fea = torch.cat([fea1_T, fea2_T, fea3_T], dim=1)
        _, score = model(fea)
    final_y = final_y.cpu().detach().numpy()
    final_output = score.cpu().detach().numpy()
    final_output = pd.DataFrame(final_output)
    final_y = pd.DataFrame(final_y)
    final1 = pd.concat([final_y, final_output], ignore_index=True,axis=1)

    final_y.columns =['Ti','Ni','Cu','Hf','Co','Zr','Pd','Ta','Nb','Al',
                           '一级处理温度/100℃','一级处理时间/h','冷却方式','冷加工温度/100℃','冷加工率/10%',
                           '二级处理温度/100℃','二级处理时间/h','冷却方式.1',
                           '应力 /100Mpa',]
    final1.columns =['Ti','Ni','Cu','Hf','Co','Zr','Pd','Ta','Nb','Al',
                           '一级处理温度/100℃','一级处理时间/h','冷却方式','冷加工温度/100℃','冷加工率/10%',
                           '二级处理温度/100℃','二级处理时间/h','冷却方式.1',
                           '应力 /100Mpa','Ms','workout']
    # final_y.to_pickle(f'sampling_{batch_size}.pkl')
    # final1.to_csv(f'sampling_{batch_size}_with_p.csv')


    real_data_path = 'data/SME3.csv'
    # 加载真实数据
    real_data = pd.read_csv(real_data_path)
    real_data.fillna(0, inplace=True)
    real_data.replace({"WQ": 1, "FC": 3, "AC": 2}, inplace=True)

    # 选择和重命名列
    columns = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Pd', 'Ta', 'Nb', 'Al',
               '一级处理温度/100℃', '一级处理时间/h', '冷却方式', '冷加工温度/100℃', '冷加工率/10%',
               '二级处理温度/100℃', '二级处理时间/h', '冷却方式.1']
    real_data = real_data.iloc[:, list(range(10)) + list(range(12, 20))]
    real_data.columns = columns
    real_data[['一级处理温度/100℃', '冷加工温度/100℃', '二级处理温度/100℃']] /= 100
    real_data[['冷加工率/10%']] /= 10
    real_data[['一级处理时间/h', '二级处理时间/h']] /= 60


    # 计算唯一性分数
    start_time = time.time()
    print(batch_size)
    print(uniqueNew_score(final_y.iloc[:, :18],real_data))
    end_time = time.time()
    unique_time = end_time-start_time
    print(unique_time)






