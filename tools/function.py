import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

def transform_c_p(fea1,fea2,fea3):

    fea1_T = fea1 / fea1.sum(dim=1, keepdim=True) * 100

    # 调整第3列，冷却方式，0-1.5变为1，1.5-2.5变为2，其余变为3
    fea2[:, 2] = torch.where((fea2[:, 2] >= 0) & (fea2[:, 2] <= 1.5), 1,
                             torch.where((fea2[:, 2] > 1.5) & (fea2[:, 2] <= 2.5), 2, 3))
    fea2[:, 0] = torch.where(fea2[:, 0] > 10.9, 10.9, fea2[:, 0])
    # 修改一级和二级热处理温度，如果低于350度，则统一赋值为350
    fea2[:, 0] = torch.where(fea2[:, 0] < 3.5, 0, fea2[:, 0])

    # 第1，2，3有一个为0值都置为0
    rows_to_zero = (fea2[:, 0] == 0) | (fea2[:, 1] == 0) | (fea2[:, 2] == 0)
    fea2[rows_to_zero, 0:3] = 0

    # 调整第8列，冷却方式
    fea2[:, 7] = torch.where((fea2[:, 7] >= 0) & (fea2[:, 7] <= 1.5), 1,
                             torch.where((fea2[:, 7] > 1.5) & (fea2[:, 7] <= 2.5), 2, 3))
    fea2[:, 5] = torch.where(fea2[:, 5] > 10.9, 10.9, fea2[:, 5])
    # 修改一级和二级热处理温度，如果低于350度，则统一赋值为350
    fea2[:, 5] = torch.where(fea2[:, 5] < 3.5, 0, fea2[:, 5])
    # 找到第7列为0的行，第6列和第8列不为0
    indices = torch.nonzero((fea2[:, 5] != 0) & (fea2[:, 7] != 0) & (fea2[:, 6] == 0))
    # 将第7列的值修改为0.08
    fea2[indices, 6] = 0.08
    # 第6，7，8有一个为0值都置为0
    rows_to_zero = (fea2[:, 5] == 0) | (fea2[:, 6] == 0) | (fea2[:, 7] == 0)
    fea2[rows_to_zero, 5:8] = 0

    # 第5列为0值，则4，5列都置为0
    rows_to_zero = (fea2[:, 4] == 0)
    fea2[rows_to_zero, 3:5] = 0
    fea2[:, 3] = torch.where(fea2[:, 3] > 10.9, 10.9, fea2[:, 3])
    fea3[:, 0] = torch.where(fea3[:, 0] == 0.0, 3, fea3[:, 0])

    return fea1_T,fea2,fea3

def data_processing(ori_data):
    # 分割数据集,前10列成分数据集，11-20工艺，21列测试应力,28列测试方法（独热）
    fea_data = ori_data.iloc[:,list(range(21)) + [28]].copy()
    # 将"WQ"替换为1，"FC"替换为3，"AC"替换为2
    fea_data.replace({"WQ": 1, "FC": 3, "AC": 2}, inplace=True)

    # 定义一个函数，用于将列中的值进行替换,0替换为1，0，0；1替换为0，1，0；2替换为0，0，1
    def replace_values(value):
        if value == 0:
            return [1, 0, 0]
        elif value == 1:
            return [0, 1, 0]
        elif value == 2:
            return [0, 0, 1]
        else:
            return value  # 如果不是0、1、2，保持原值
    # 应用函数来替换指定列的值
    fea_data['method'] = fea_data['method'].apply(replace_values)
    # 将某一列的list拆分为三列
    fea_data[['col1', 'col2', 'col3']] = fea_data['method'].apply(pd.Series)
    # 删除原始列
    fea_data = fea_data.drop('method', axis=1)
    fea_data.iloc[:,[10,12,15,17,20]] = fea_data.iloc[:,[10,12,15,17,20]]/100
    fea_data.iloc[:, [11,16]] = fea_data.iloc[:, [11,16]] / 10
    fea_data.iloc[:, [13, 18]] = fea_data.iloc[:, [13, 18]] / 60
    properties_data = ori_data.loc[:,['Ms/℃','workout J/cm3']].copy()

    # 合并前10列为 list1
    list1 = fea_data.iloc[:, :10].values.tolist()
    # 合并中间10列为 list2
    list2 = fea_data.iloc[:, 12:20].values.tolist()
    # 合并最后2列为 list3
    list3 = fea_data.iloc[:, 20:].values.tolist()
    list4 = properties_data.iloc[:,:].values.tolist()
    # 创建新的 DataFrame
    fea_prop_data = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})

    return fea_prop_data


def new_split(fea_prop_data,ratio,rm):
    # ratio训练集和测试集的比例，e.g. 0.2 —— 训练集80%测试集20%
    # rm: 随机数种子
    fea_prop_data_new = pd.DataFrame()
    # 将某一列的list拆分为三列
    fea_prop_data_new[[i for i in range(10)]] = fea_prop_data['list1'].apply(pd.Series)
    fea_prop_data_new[[i+10 for i in range(10)]] = fea_prop_data['list2'].apply(pd.Series)
    fea_prop_data_new[[i+20 for i in range(4)]] = fea_prop_data['list3'].apply(pd.Series)
    fea_prop_data_new[[i+24 for i in range(2)]] = fea_prop_data['list4'].apply(pd.Series)

    # 1. 选择前20列得到dataframe1
    dataframe1 = fea_prop_data_new.iloc[:, :20]

    # 2. dataframe1去除重复
    dataframe1 = dataframe1.drop_duplicates()

    # 3. 8:2划分训练集和测试集
    train_data, test_data = train_test_split(dataframe1, test_size=ratio, random_state=rm, shuffle=True)

    # 4. 将dataframe中前20列和训练集相同的组成最终的训练集
    final_train_set = fea_prop_data_new[fea_prop_data_new.iloc[:, :20].isin(train_data.to_dict('list')).all(axis=1)]

    # 5. dataframe中其余元素组成测试集
    final_test_set = fea_prop_data_new[~fea_prop_data_new.index.isin(final_train_set.index)]

    #final_test_set
    # 合并前10列为 list1
    list1 = final_train_set.iloc[:, :10].values.tolist()
    # 合并中间10列为 list2
    list2 = final_train_set.iloc[:, 10:20].values.tolist()
    # 合并最后2列为 list3
    list3 = final_train_set.iloc[:, 20:24].values.tolist()
    list4 = final_train_set.iloc[:, 24:26].values.tolist()
    # 创建新的 DataFrame
    final_train_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})

    #final_test_set
    # 合并前10列为 list1
    list1 = final_test_set.iloc[:, :10].values.tolist()
    # 合并中间10列为 list2
    list2 = final_test_set.iloc[:, 10:20].values.tolist()
    # 合并最后2列为 list3
    list3 = final_test_set.iloc[:, 20:24].values.tolist()
    list4 = final_test_set.iloc[:, 24:26].values.tolist()
    # 创建新的 DataFrame
    final_test_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})

    return final_train_list,final_test_list


def new_split_kf(fea_prop_data,ratio,rm):
    # ratio训练集和测试集的比例，e.g. 10 —— 10折交叉验证
    # rm: 随机数种子
    fea_prop_data_new = pd.DataFrame()
    # 将某一列的list拆分为三列
    fea_prop_data_new[[i for i in range(10)]] = fea_prop_data['list1'].apply(pd.Series)
    fea_prop_data_new[[i+10 for i in range(10)]] = fea_prop_data['list2'].apply(pd.Series)
    fea_prop_data_new[[i+20 for i in range(4)]] = fea_prop_data['list3'].apply(pd.Series)
    fea_prop_data_new[[i+24 for i in range(2)]] = fea_prop_data['list4'].apply(pd.Series)
    dataframe_Al = fea_prop_data_new[fea_prop_data_new.iloc[:,9] != 0].copy()
    dataframe_NoAl = fea_prop_data_new[fea_prop_data_new.iloc[:, 9] == 0].copy()
    # 1. 选择前20列得到dataframe1
    dataframe1 = dataframe_NoAl.iloc[:, :20]

    # 2. dataframe1去除重复
    dataframe1 = dataframe1.drop_duplicates()

    # 3. 10折交叉验证
    kf = KFold(n_splits=ratio, random_state=rm, shuffle=True)
    # 采用list存储十折交叉产生的训练集和测试集
    final_train_list_kf, final_test_list_kf = [], []
    for train_index, test_index in kf.split(dataframe1):
        train_data, test_data = dataframe1.iloc[train_index], dataframe1.iloc[test_index]
        # 4. 将dataframe中前20列和训练集相同的组成最终的训练集
        final_train_set = fea_prop_data_new[fea_prop_data_new.iloc[:, :20].isin(train_data.to_dict('list')).all(axis=1)]
        final_train_set = pd.concat([final_train_set,dataframe_Al])
        # 5. dataframe中其余元素组成测试集
        final_test_set = fea_prop_data_new[~fea_prop_data_new.index.isin(final_train_set.index)]

        #final_test_set
        # 合并前10列为 list1
        list1 = final_train_set.iloc[:, :10].values.tolist()
        # 合并中间10列为 list2
        list2 = final_train_set.iloc[:, 10:20].values.tolist()
        # 合并最后2列为 list3
        list3 = final_train_set.iloc[:, 20:24].values.tolist()
        list4 = final_train_set.iloc[:, 24:26].values.tolist()
        # 创建新的 DataFrame
        final_train_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})

        #final_test_set
        # 合并前10列为 list1
        list1 = final_test_set.iloc[:, :10].values.tolist()
        # 合并中间10列为 list2
        list2 = final_test_set.iloc[:, 10:20].values.tolist()
        # 合并最后2列为 list3
        list3 = final_test_set.iloc[:, 20:24].values.tolist()
        list4 = final_test_set.iloc[:, 24:26].values.tolist()
        # 创建新的 DataFrame
        final_test_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})
        final_train_list_kf.append(final_train_list)
        final_test_list_kf.append(final_test_list)

    return final_train_list_kf,final_test_list_kf

def new_split_kf2(fea_prop_data,ratio,rm):
    # 删除热加工工艺的十折交叉验证
    # ratio训练集和测试集的比例，e.g. 10 —— 10折交叉验证
    # rm: 随机数种子
    fea_prop_data_new = pd.DataFrame()
    # 将某一列的list拆分为三列
    fea_prop_data_new[[i for i in range(10)]] = fea_prop_data['list1'].apply(pd.Series)
    fea_prop_data_new[[i+10 for i in range(8)]] = fea_prop_data['list2'].apply(pd.Series)
    fea_prop_data_new[[i+18 for i in range(4)]] = fea_prop_data['list3'].apply(pd.Series)
    fea_prop_data_new[[i+22 for i in range(2)]] = fea_prop_data['list4'].apply(pd.Series)
    dataframe_Al = fea_prop_data_new[fea_prop_data_new.iloc[:,9] != 0].copy()
    dataframe_NoAl = fea_prop_data_new[fea_prop_data_new.iloc[:, 9] == 0].copy()
    # 1. 选择前20列得到dataframe1
    dataframe1 = dataframe_NoAl.iloc[:, :18]

    # 2. dataframe1去除重复
    dataframe1 = dataframe1.drop_duplicates()

    # 3. 10折交叉验证
    kf = KFold(n_splits=ratio, random_state=rm, shuffle=True)
    # 采用list存储十折交叉产生的训练集和测试集
    final_train_list_kf, final_test_list_kf = [], []
    for train_index, test_index in kf.split(dataframe1):
        train_data, test_data = dataframe1.iloc[train_index], dataframe1.iloc[test_index]
        # 4. 将dataframe中前18列和训练集相同的组成最终的训练集
        # 提取前18列
        fea_prop_data_new_subset = fea_prop_data_new.iloc[:, :18]
        # 找到完全相同的行
        merged_df = pd.merge(train_data, fea_prop_data_new_subset, on=list(train_data.columns), how='inner')
        final_train_set = fea_prop_data_new[fea_prop_data_new.iloc[:, :18].apply(tuple, axis=1).isin(merged_df.apply(tuple, axis=1))]
        # train_data_dict = train_data.to_dict('list')
        # final_train_set = fea_prop_data_new[fea_prop_data_new.iloc[:, :18].isin(train_data.to_dict('list')).all(axis=1)]
        final_train_set = pd.concat([final_train_set,dataframe_Al])
        # 5. dataframe中其余元素组成测试集
        final_test_set = fea_prop_data_new[~fea_prop_data_new.index.isin(final_train_set.index)]

        #final_test_set
        # 合并前10列为 list1
        list1 = final_train_set.iloc[:, :10].values.tolist()
        # 合并中间10列为 list2
        list2 = final_train_set.iloc[:, 10:18].values.tolist()
        # 合并最后2列为 list3
        list3 = final_train_set.iloc[:, 18:22].values.tolist()
        list4 = final_train_set.iloc[:, 22:24].values.tolist()
        # 创建新的 DataFrame
        final_train_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})

        #final_test_set
        # 合并前10列为 list1
        list1 = final_test_set.iloc[:, :10].values.tolist()
        # 合并中间10列为 list2
        list2 = final_test_set.iloc[:, 10:18].values.tolist()
        # 合并最后2列为 list3
        list3 = final_test_set.iloc[:, 18:22].values.tolist()
        list4 = final_test_set.iloc[:, 22:24].values.tolist()
        # 创建新的 DataFrame
        final_test_list = pd.DataFrame({'list1': list1, 'list2': list2, 'list3': list3, 'list4': list4})
        final_train_list_kf.append(final_train_list)
        final_test_list_kf.append(final_test_list)

    return final_train_list_kf,final_test_list_kf


