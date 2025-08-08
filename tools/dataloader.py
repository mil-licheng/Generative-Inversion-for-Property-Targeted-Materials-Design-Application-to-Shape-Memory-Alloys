from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

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
        fea123 = self.landmarks_frame.iloc[idx,4]
        fea123 = np.array(fea123)
        fea123 = fea123.astype(np.float32)
        properties = self.landmarks_frame.iloc[idx,3]
        properties = np.array(properties)
        properties = properties.astype(np.float32)
        sample = {'fea123': fea123, 'properties': properties}
        if self.transform:
            sample = self.transform(sample)

        return sample