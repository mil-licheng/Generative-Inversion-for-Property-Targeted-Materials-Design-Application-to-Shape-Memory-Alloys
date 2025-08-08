"""
自定义评判生成分数
统计dataframe中前10列加和不为100的行数
热加工温度，第11列，大于10.9
热加工率，第12列，大于9
第12列为0，第11不为0
一级处理温度，时间，冷却方式，第13、14、15列，不同时为0或者非0
冷加工温度，第16列非0，第17为0
二级处理温度，时间，冷却方式，第18、19、20列，不同时为0或者非0
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KDTree
from sklearn.preprocessing import normalize

def get_score1(df):
    # 统计前10列加和不为100的行数
    rows_not_sum_100 = df[(df.iloc[:, :10].sum(axis=1) != 100)]
    # 第11列大于10.9
    col11_gt_10_9 = df[df.iloc[:, 10] > 10.9]

    # 第12列大于9
    col12_gt_9 = df[df.iloc[:, 11] > 9]

    # 第12列为0，第11不为0
    col12_eq_0_col11_not_eq_0 = df[(df.iloc[:, 11] == 0) & (df.iloc[:, 10] != 0)]

    # 第13、14、15列，不同时为0或者非0
    col13_14_15_not_eq_0 = df[(df.iloc[:, 12] != 0) & (df.iloc[:, 13] != 0) & (df.iloc[:, 14] != 0)]
    col13_14_15_all_0 = df[(df.iloc[:, 12] == 0) & (df.iloc[:, 13] == 0) & (df.iloc[:, 14] == 0)]
    col13_14_15 = len(df) - len(col13_14_15_not_eq_0) - len(col13_14_15_all_0)


    # 第16列非0，第17为0
    col16_not_eq_0_col17_eq_0 = df[(df.iloc[:, 15] != 0) & (df.iloc[:, 16] == 0)]

    # 第18、19、20列，不同时为0或者非0
    col18_19_20_not_eq_0 = df[(df.iloc[:, 17] != 0) & (df.iloc[:, 18] != 0) & (df.iloc[:, 19] != 0)]
    col18_19_20_all_0 = df[(df.iloc[:, 17] == 0) & (df.iloc[:, 18] == 0) & (df.iloc[:, 19] == 0)]
    col18_19_20 = len(df) - len(col18_19_20_not_eq_0) - len(col18_19_20_all_0)


    # score1 = (
    #         len(rows_not_sum_100) + len(col11_gt_10_9) + len(col12_gt_9) + len(col12_eq_0_col11_not_eq_0) +
    #         col13_14_15 + len(col16_not_eq_0_col17_eq_0) + col18_19_20
    # )/len(df)

    score1 = (
            0 + len(col11_gt_10_9) + len(col12_gt_9) + len(col12_eq_0_col11_not_eq_0) +
            col13_14_15 + len(col16_not_eq_0_col17_eq_0) + col18_19_20
    )/len(df)

    return score1

def get_score1_2(df):
    # 删除热加工工艺后的惩罚分数计算
    # 统计前10列加和不为100的行数
    rows_not_sum_100 = df[(df.iloc[:, :10].sum(axis=1) != 100)]

    # 第11、12、13列，不同时为0或者非0
    col11_12_13_not_eq_0 = df[(df.iloc[:, 10] != 0) & (df.iloc[:, 11] != 0) & (df.iloc[:, 12] != 0)]
    col11_12_13_all_0 = df[(df.iloc[:, 10] == 0) & (df.iloc[:, 11] == 0) & (df.iloc[:, 12] == 0)]
    col11_12_13 = len(df) - len(col11_12_13_not_eq_0) - len(col11_12_13_all_0)


    # 第14列非0，第15为0
    col14_not_eq_0_col15_eq_0 = df[(df.iloc[:, 13] != 0) & (df.iloc[:, 14] == 0)]

    # 第16、17、18列，不同时为0或者非0
    col16_17_18_not_eq_0 = df[(df.iloc[:, 15] != 0) & (df.iloc[:, 16] != 0) & (df.iloc[:, 17] != 0)]
    col16_17_18_all_0 = df[(df.iloc[:, 15] == 0) & (df.iloc[:, 16] == 0) & (df.iloc[:, 17] == 0)]
    col16_17_18 = len(df) - len(col16_17_18_not_eq_0) - len(col16_17_18_all_0)


    score1 = (
            0 + col11_12_13 + len(col14_not_eq_0_col15_eq_0) + col16_17_18
    )/len(df)

    return score1
# def get_score2(df):
#     score2 = 0
#     # 统计每行非0数值的个数，并存储在新的列中
#     df['non_zero_count'] = df.apply(lambda row: (row != 0).sum(), axis=1)
#     for i in range(3,len(df.columns)):
#         count_rows = (df['non_zero_count'] == i).sum()
#         score2 = score2 + count_rows*i
#     score2 = score2/len(df)
#     z = len(df['non_zero_count'].value_counts())
#     return score2*np.log10(z)
def get_score2(df):

    similarities = cosine_similarity(df)

    # 设置相似度阈值
    threshold = 0.999999
    # 寻找相似行并删除
    duplicates = []
    for i in range(len(similarities)):
        for j in range(i + 1, len(similarities)):
            if similarities[i][j] > threshold:
                duplicates.append(j)
    # 删除相似行
    df_want = df.drop(duplicates)
    zero_columns = [col for col in df.columns if (df[col] == 0).all()]
    score2 = len(df_want)/len(df) - len(zero_columns)

    return score2



def uniqueNew_score(df,df_train):
    n = df.shape[0]
    df_unique = unique_sample(df)
    unique_score = len(df_unique) / n
    df_unique_new = uniqueNew_sample(df_unique,df_train)
    unique_new_len = len(df_unique_new)
    unique_new_score = unique_new_len / n

    return unique_score,unique_new_score


def unique_sample(df):
    # 设置相似度阈值
    threshold = 0.999999
    n = df.shape[0]
    sample_number = 10**5
    if n > sample_number:
        # sample_frequency = int(n//sample_number)*2
        sample_frequency = 10
        for i in range(sample_frequency):
            sampled_df = df.sample(n=sample_number, random_state=0+i)
            duplicate_indices = _unique_sample(sampled_df, threshold)
            df = df.drop(duplicate_indices)
            df.reset_index(inplace=True, drop=True)
        df_unique = df
    else:
        duplicate_indices = _unique_sample(df,threshold)
        # 删除重复行
        df_unique = df.drop(duplicate_indices)

    return df_unique


def _unique_sample(df,threshold):
    # 预计算余弦相似度矩阵
    similarities = cosine_similarity(df)
    n = similarities.shape[0]

    # 获取上三角矩阵索引（排除对角线）
    i, j = np.triu_indices(n, k=1)

    # 向量化比较并获取重复行索引
    dup_mask = similarities[i, j] > threshold
    duplicate_indices = np.unique(j[dup_mask])  # 保证索引唯一性

    return duplicate_indices


def uniqueNew_sample(df_unique,df_train):
    # 计算整个批次与当前df_want的相似度
    similarities = cosine_similarity(df_unique, df_train)
    max_similarities = np.max(similarities, axis=1)

    threshold = 0.9995
    # 筛选出不相似的行
    mask = max_similarities < threshold
    df_unique_new = df_unique[mask]

    return df_unique_new






if __name__ == '__main__':
    np.random.seed(42)
    # 生成一个26列10000行的矩阵
    matrix = np.random.rand(10000, 26)
    columns = [f'Col{i + 1}' for i in range(26)]
    df = pd.DataFrame(matrix, columns=columns)
    score1 = get_score1(df)
    print(score1)

    score2 = get_score2(df)
    print(score2)

