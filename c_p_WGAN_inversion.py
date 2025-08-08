import torch
import torch.nn as nn
from torch.autograd import Variable
import pandas as pd
import numpy as np
import time
from networks.ann_predict import PFDataset,CustomLoss,evalute,MyModel,MyModel2,MyModel3
from networks.c_p_WGAN_gp import Generator,Discriminator
from tools.function import transform_c_p
from sklearn.metrics.pairwise import cosine_similarity



class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, prediction,target,type="Ms_workout"):
        mse_loss = nn.MSELoss()
        prediction[torch.isnan(prediction)] = 0.0
        loss_col1 = mse_loss(prediction[:, 0], target[:, 0])
        loss_col2 = mse_loss(prediction[:, 1], target[:, 1])
        if type == "Ms_workout":
            total_loss = loss_col1 + 5 * loss_col2
        elif type == "Ms":
            total_loss = loss_col1
        elif type == "workout":
            total_loss = loss_col2
        return total_loss


if __name__ == '__main__':
    #设置随机数种子
    seed = 0
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
    generator.load_state_dict(
        torch.load("H:\\licheng\\ML_HTSMAs\\code\\GAN\\model\\generator4096_94501_10_2_im_14_2.pth"))
    # generator.load_state_dict(
    #     torch.load("/share/home/3121102080/SME/generator4096_94501_10_2_im_14_2.pth"))

    hidden_size = 512
    iter = 1
    layer_numbers = 2
    save_path = "H:/licheng/ML_HTSMAs/code/v2/ann_{}_{}_{}_3_1.pth".format(iter+1,hidden_size,layer_numbers) # 模型权重参数保存位置
    # save_path = "/share/home/3121102080/SME/ann_{}_{}_{}_3_1.pth".format(iter+1,hidden_size,layer_numbers) # 模型权重参数保存位置

    model = MyModel3(input_size1=14, input_size2=8, hidden_size=hidden_size, layer_numbers=layer_numbers,
                     layer_numbers0=1,sub_number=8)
    model.load_state_dict(torch.load(save_path))
    model.to(device)


    loss_list = []
    number_list = []
    epoch_list = []
    batch_size, input_size = 1000, input_size_G
    # 输入向量 z 的初始化
    z = torch.randn(batch_size, input_size, device=device,requires_grad=True)
    # 优化器
    # optimizer = torch.optim.Adam([z], lr=0.04)
    optimizer = torch.optim.Adam([z], lr=0.04)
    # 初始化表格站位
    data = [0] * 31  # 假设数据都初始化为 0

    # 存储隐空间z的变量
    z_dataframe = pd.DataFrame()
    # 目标
    ms,workout = 0.0,30.0
    target = torch.tensor([[ms, workout]],device=device)
    # 使用expand操作扩展为batch_sizex2的张量
    target = target.expand(batch_size, -1)
    # target = Variable(target, requires_grad=False)

    ways = torch.tensor([[1, 0, 0]],device=device)
    ways = ways.expand(batch_size, -1)
    # 迭代寻找满足目标的成分和工艺
    num_epochs = 6000
    start_time = time.time()
    for epoch in range(num_epochs):
        # 通过生成器生成样本 y
        generated_output = generator(z)
        z_data = z.cpu().detach().numpy()

        # 通过评估网络计算Ms和workout
        fea1 = generated_output[:, :10].clone()
        fea2 = generated_output[:, 10:18].clone()
        fea3 = generated_output[:, 18:].clone()

        fea1_T, fea2_T, fea3_T = transform_c_p(fea1, fea2, fea3)
        # fea1_T, fea2_T, fea3_T = fea1, fea2, fea3

        final_y = torch.cat([fea1_T, fea2_T, fea3_T], dim=1)
        fea3_T = torch.cat((fea3_T, ways), dim=1)
        fea = torch.cat([fea1_T, fea2_T, fea3_T], dim=1)
        _,score = model(fea)
        # score = model(fea1_T, fea2_T, fea3_T)
        #score = model(fea1,fea2,fea3)

        final_y = final_y.cpu().detach().numpy()
        final_output = score.cpu().detach().numpy()
        final_y = pd.DataFrame(final_y)
        final_output = pd.DataFrame(final_output)
        z_data = pd.DataFrame(z_data)
        final1 = pd.concat([final_y, final_output,z_data], ignore_index=True,axis=1)
        final1.columns =['Ti','Ni','Cu','Hf','Co','Zr','Pd','Ta','Nb','Al',
                               '一级处理温度/100℃','一级处理时间/h','冷却方式','冷加工温度/100℃','冷加工率/10%',
                               '二级处理温度/100℃','二级处理时间/h','冷却方式.1',
                               '应力 /100Mpa','Ms','workout'] + list(range(10))

        # 计算目标函数
        loss_fun = CustomLoss()
        loss = loss_fun(score, target,type="Ms")
        # output：loss，input：loss
        grad_fea1 = torch.autograd.grad(loss, fea1_T,retain_graph=True)[0]
        grad_fea2 = torch.autograd.grad(loss, fea2_T,retain_graph=True)[0]
        grad_fea3 = torch.autograd.grad(loss, fea3_T,retain_graph=True)[0]

        grad_fea = torch.cat((grad_fea1, grad_fea2, grad_fea3[:,[0]]), dim=1)
        # grad_fea = torch.randn(batch_size, 26, device=device, requires_grad=True)
        # print(z.grad)
        # 反向传播，计算梯度
        optimizer.zero_grad()
        # loss.backward()
        generated_output.backward(grad_fea,retain_graph=True)
        # 更新输入 z，以最小化目标函数
        optimizer.step()
        if epoch == 0:
            final1.to_csv('final_want_{}_{}_{}.csv'.format(ms, workout, epoch))
            loss_list.append(loss.item())
            number_list.append(0)
        # 打印训练过程中的信息
        elif (epoch + 1) % 100 == 0:
            final_iter = final1[
                (final1['Ms'] > ms - 10) & (final1['Ms'] < ms + 10) & (final1['workout'] > workout - 2) & (
                            final1['workout'] < workout + 2)]
            final_iter = np.around(final_iter, decimals=1)
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
            loss_list.append(loss.item())
            final_iter.reset_index(inplace=True, drop=True)

            final_iter_noz = final_iter.iloc[:, :21].copy()
            # similarities = cosine_similarity(final_iter_noz)
            #
            # # 设置相似度阈值
            # threshold = 0.999999
            #
            # # 寻找相似行并删除
            # duplicates = []
            # for i in range(len(similarities)):
            #     for j in range(i + 1, len(similarities)):
            #         if similarities[i][j] > threshold:
            #             duplicates.append(j)
            #
            # # 删除相似行
            # final_iter = final_iter.drop(duplicates)
            # final_iter = final_iter.sort_values(by='应力 /100Mpa', ascending=True)
            # number_list.append(final_iter.shape[0])
            if (epoch + 1) % 1000 == 0:
                final1.to_csv('final_want_{}_{}_{}.csv'.format(ms, workout, epoch+1))

    data_loss_number = pd.DataFrame({'loss':loss_list,'number':number_list})
    data_loss_number.to_csv('data_loss_number.csv')





