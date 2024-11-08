import torch
print(1)
#环境准备
import numpy as np              # numpy数组库
import math                     # 数学运算库
import matplotlib.pyplot as plt # 画图库
import time as time
import pandas as pd
import torch             # torch基础库
import torch.nn as nn    # torch神经网络库
import torch.nn.functional as F
import torchvision.transforms as transforms  #公开数据集的预处理库,格式转换
import torchvision.utils as utils
import torch.utils.data as data_utils  #对数据集进行分批加载的工具集
from PIL import Image #图片显示
from collections import OrderedDict
import torchvision.models as models
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
from torch.autograd import Variable
from data_load_vit import load_data,load_data_val,load_data_new,load_data_3
import os
from sklearn import metrics

print("Hello World")
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.version())

numpy_path = '/home/hpc/users/hanyuzhe/胆囊_vit/np'
batch_size = 10

train_transform = transforms.Compose([
    transforms.Grayscale(3),
    transforms.RandomResizedCrop(224), #随机裁剪一个area然后再resize
    transforms.RandomHorizontalFlip(), #随机水平翻转
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class subDataset(Dataset.Dataset):
    # 初始化，定义数据内容和标签
    def __init__(self, Data, Label, Cli):
        self.Data = torch.tensor(Data, dtype=torch.float)
        self.Label = torch.tensor(Label)
        ### 这里的Cli 已经是tensor
        # self.Cli = Cli
        ## 如果是numpy
        self.Cli= torch.tensor(Cli)

    # 返回数据集大小
    def __len__(self):
        return len(self.Data)

    # 得到数据内容和标签
    def __getitem__(self, index):
        data = self.Data[index]

        label = self.Label[index]

        cli = self.Cli[index]

        return data, label, cli

# CT_data_path = '/home/wq/gallbladder/cancer/merge_cancer_two'
# df = pd.read_csv('/home/hyz/tmp/pycharm_project_519/胆囊癌/训练_clean.csv',encoding='utf-8')
new_path ='/home/hpc/users/hanyuzhe/胆囊_vit/最新胆囊数据'
new_path_3 = '/home/hpc/users/hanyuzhe/胆囊_vit/胆囊数据第三次/image第三次'
CT_data_path = '/home/hpc/users/wangqing/gallbladder/vc_data/cancer/merge_cancer'
val_data_path = '/home/hpc/users/wangqing/gallbladder/gallbladder/val/merge_cancer'
df = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/训练_clean.csv')
df_val = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/验证_clean.csv')

X,Y,clinical = load_data(CT_data_path,df)
val_X,val_Y,val_clinical = load_data_val(val_data_path,df_val)

X1, Y1, clinical1, X2, Y2, clinical2 = load_data_new(new_path)
X1_3, Y1_3, clinical1_3, X2_3, Y2_3, clinical2_3 = load_data_3(new_path_3)

clinical_number = 20
###
def manage_split(x,y,cli):
    ## 临床数据的选择
    cli = cli[:,:clinical_number]
    val_length = x.shape[0] // 5
    test_x = x[:val_length]
    train_x = x[val_length:]

    test_x = test_x.swapaxes(1, 3)
    train_x = train_x.swapaxes(1, 3)

    test_y = y[:val_length]
    train_y = y[val_length:]

    test_clinical = cli[:val_length]
    train_clinical = cli[val_length:]

    print('train_x', train_x.shape)
    print('test_x', test_x.shape)


    return train_x,train_y,train_clinical,test_x,test_y,test_clinical

def manage_val(x,y,cli):
    val_clinical = cli[:, :clinical_number]
    ########
    val_x = x.swapaxes(1, 3)
    val_y = y
    print('========val.shape========')
    print(val_x.shape)
    print(val_y.shape)
    print(val_clinical.shape)
    return val_x,val_y,val_clinical

train_x,train_y,train_clinical,test_x,test_y,test_clinical = manage_split(X,Y,clinical)
val_x,val_y,val_clinical = manage_val(val_X,val_Y,val_clinical)
print('train_x',train_x.shape)
print('test_x',test_x.shape)
print('val_x',val_x.shape)

train_dataset = subDataset(train_x, train_y, train_clinical)
train_dataloader = DataLoader.DataLoader(train_dataset, batch_size=10, shuffle=False, num_workers=0)


for batch_images, targets,batch_cli in train_dataloader:
    print(batch_images.shape)
    print(targets.shape)
    print(batch_cli.shape)
    break

test_dataset = subDataset(test_x, test_y, test_clinical)
test_dataloader = DataLoader.DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=0)

val_dataset = subDataset(val_x, val_y, val_clinical)
val_dataloader = DataLoader.DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=0)
