import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tools.score import get_score1,get_score2,get_score1_2
import argparse
import random
from tools.function import data_processing
from tools.dataloader import PFDataset

# 定义生成器和判别器
class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_numbers):
        super(Generator, self).__init__()
        model = []
        model.append(nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        ))
        for i in range(layer_numbers):
            model.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.ReLU()
            ))
            hidden_size = hidden_size//2
        model.append(nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.ReLU()
        ))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_numbers):
        super(Discriminator, self).__init__()
        model = []
        model.append(nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2)
        ))

        for i in range(layer_numbers):
            model.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size//2),
                nn.LeakyReLU(0.2)
            ))
            hidden_size = hidden_size//2

        model.append(nn.Sequential(
            nn.Linear(hidden_size, output_size),
        ))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)



# 打印生成器和判别器的平均梯度
def print_average_gradient(model, name):
    total_parameters = 0
    total_gradients = 0

    for param in model.parameters():
        if param.requires_grad:
            total_parameters += param.numel()
            total_gradients += param.grad.abs().sum()

    average_gradient = total_gradients / total_parameters
    print(f"{name} 平均梯度: {average_gradient.item()}")


if __name__ == '__main__':
    # 设置命令行输入
    parser = argparse.ArgumentParser()
    parser.description = 'please enter parameters：num_epochs'
    parser.add_argument("-hidden_size", "--hidden_size", help="this is parameter hidden_size", dest="hidden_size", type=int
                        , default="4096")
    parser.add_argument("-Lambda", "--Lambda", help="this is parameter Lambda", dest="Lambda", type=int
                        , default="10")
    parser.add_argument("-layers", "--layers", help="this is parameter layers", dest="layers", type=int
                        , default="2")
    args = parser.parse_args()

    #设置随机数种子
    seed = 1008
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # 定义超参数
    input_size_G,input_size_D = 10,19  # 输入噪声的维度
    hidden_size = args.hidden_size  # 隐藏层维度
    layer_numbers = args.layers
    output_size_G,output_size_D = 19,1  # 输出维度
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # 创建生成器和判别器实例
    generator = Generator(input_size_G, hidden_size, output_size_G, layer_numbers).to(device)
    discriminator = Discriminator(input_size_D, hidden_size, output_size_D, layer_numbers).to(device)

    # generator.load_state_dict(torch.load("/share/home/3121102080/SME/generator102.pth"))
    # discriminator.load_state_dict(torch.load("/share/home/3121102080/SME/discriminator102.pth"))
    inum = 0
    iter = 1
    # 训练GAN
    num_epochs = 120000
    batch_size = 75

    # 读取文件，数据预处理
    path = '/share/home/3121102080/SME/'
    # path = 'H:/licheng/ML_HTSMAs/'
    ori_path = path + 'SME3.csv'
    ori_data = pd.read_csv(filepath_or_buffer=ori_path)
    ori_data = ori_data.fillna(0)
    fea_prop_data = data_processing(ori_data)

    fea3 = fea_prop_data.loc[:, ['list3']]
    fea3_1 = fea3['list3'].apply(pd.Series)
    fea3_1 = fea3_1.drop([1,2,3], axis=1)
    fea_prop_data['list3'] = fea3_1.apply(lambda row: row.tolist(), axis=1)

    fea_prop_data['list123'] = fea_prop_data.apply(lambda row: row['list1'] + row['list2'] + row['list3'], axis=1)
    train_set = PFDataset(fea_prop_data)
    trainloader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True,
                                              num_workers=1, pin_memory=True)

    # 定义优化器和损失函数
    lr = 0.0002
    optimizer_G = optim.Adam(generator.parameters(), lr=lr)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr*4)
    Lambda = args.Lambda
    Score,Score1,Score2,best_score = [],[],[], 0
    print('Lambda:{}'.format(Lambda))
    print('nz:{}'.format(input_size_G))
    for epoch in tqdm(range(num_epochs)):
        for i, data in enumerate(trainloader, 0):
            # 生成真实数据和噪声数据
            real_data = data['fea123'].to(device)

            fake_data = generator(torch.randn(batch_size, input_size_G, device=device))

            # 训练判别器
            optimizer_D.zero_grad()
            output_real = discriminator(real_data)
            output_fake = discriminator(fake_data.detach())
            # 计算梯度惩罚项
            alpha = torch.rand(real_data.size(0), 1).to(device)
            interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)
            d_interpolates = discriminator(interpolates)
            gradients = torch.autograd.grad(outputs=d_interpolates, inputs=interpolates,
                                            grad_outputs=torch.ones(d_interpolates.size()).to(device),
                                            create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradient_penalty = Lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            loss_D = output_fake.mean() - output_real.mean() + gradient_penalty
            # 反向传播和优化
            loss_D.backward()
            optimizer_D.step()

            # 训练生成器
            optimizer_G.zero_grad()
            noise = torch.randn(real_data.size(0), input_size_G).to(device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data)
            loss_G = -output_fake.mean()

            # 反向传播和优化
            loss_G.backward()
            optimizer_G.step()

        if epoch % 500 == 1:
            generator.eval()
            # 生成一些样本并可视化
            with torch.no_grad():
                test_samples = generator(torch.randn(10000, input_size_G, device=device)).cpu().numpy()
            # 计算生成的分数
            test_samples = pd.DataFrame(test_samples)
            score1 = -1 * get_score1_2(test_samples)
            score2 = get_score2(test_samples)
            score = score1 + score2
            Score.append(score)
            Score1.append(score1)
            Score2.append(score2)

            # # 打印训练信息
            # if (epoch+1) % 10000 == 0:
            #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}')
            #     print_average_gradient(generator, "生成器")
            #     print_average_gradient(discriminator, "判别器")
            if score > best_score:
                best_score = score
                generator_path = "/share/home/3121102080/SME/generator{}_{}_{}_{}_im_{}4_2.pth".format(hidden_size,epoch,Lambda,layer_numbers,iter)
                discriminator_path = "/share/home/3121102080/SME/discriminator{}_{}_{}_{}_im_{}4_2.pth".format(hidden_size,
                                                                                                     epoch, Lambda,
                                                                                                     layer_numbers,
                                                                                                     iter)
                # discriminator_path = "/share/home/3121102080/SME/discriminator{}_{}.pth".format(input_size_G,epoch)
                torch.save(generator.state_dict(), generator_path)
                torch.save(discriminator.state_dict(), discriminator_path)

            generator.train()

    # 创建DataFrame
    df = pd.DataFrame({'score': Score, 'score1': Score1, 'score2': Score2})
    # 将DataFrame保存为.pkl文件
    df.to_pickle('scores_z{}_{}_{}_im4_2.pkl'.format(hidden_size,Lambda,layer_numbers))