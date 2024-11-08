import torch
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

from data_load_vit import load_data,load_data_val,load_data_new
import os
from sklearn import metrics

v = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
# Create a new Linear layer with 2 output features
new_head = torch.nn.Linear(in_features=192, out_features=1024, bias=True)
# Replace the head layer of the model with the new Linear layer
v.head = new_head

clinical_number = 20
## 后面的全连接网络 创建一个net类
class net(nn.Module):
    def __init__(self):
        super(net,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1024+clinical_number, 512),   ### 1048改为1046 改1044
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    def forward(self,z,cli):
        catted   = torch.cat((z,cli),dim=1)
        return self.model(catted)
net = net()

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

net_test_pthfile = '/home/hpc/users/hanyuzhe/胆囊_vit/np/net_test_23.pth'
v_test_pthfile = '/home/hpc/users/hanyuzhe/胆囊_vit/np/v_test_23.pth'

net_val_pthfile = '/home/hpc/users/hanyuzhe/胆囊_vit/np/net_val_66.pth'
v_val_pthfile = '/home/hpc/users/hanyuzhe/胆囊_vit/np/v_val_66.pth'


X,Y,clinical = load_data(CT_data_path,df)
val_X,val_Y,val_clinical = load_data_val(val_data_path,df_val)
## 相关性分析
a=0
b=0
c=0
d=0
for i in range(Y.shape[0]):
    if Y[i,1]==1 and clinical[i,12]==1:
        a+=1
    elif Y[i,1]==1 and clinical[i,12]==0:
        b+=1
    elif Y[i, 1] == 0 and clinical[i, 12] == 1:
        c+=1
    elif Y[i,1]==0 and clinical[i,12]==0:
        d+=1
print('a,b,c,d',a,b,c,d)


# 黄色肉芽肿数量

# print('黄色肉芽肿',sum(Y[:,0]))
# print('胆囊癌',sum(Y[:,1]))

# print(sum(clinical[:,12]))
# print(sum(clinical[:,13]))


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
def accuracy_score(x,y,clinical):
    x = torch.tensor(x,dtype=torch.float)
    y = torch.tensor(y)
    clinical = torch.tensor(clinical)

    z = v(x)
    outputs = net(z, clinical)
    ret, predicted = torch.max(outputs.data, 1)
    number = y.size()[0]  # 图片的个数

    correct_number = (predicted == y[:, 1]).sum().item()  # 预测正确的数目
    accuracy = 100 * correct_number / number
    print('accuracy',accuracy)
    return(accuracy)
def permutation_importance(x, y,clinical, n_permutations=10):
    baseline_score = accuracy_score(x,y,clinical)
    importances = []

    num_vars = clinical.shape[1] // 2

    for i in range(num_vars):
        permuted_clinical = clinical.copy()
        indices = np.arange(permuted_clinical.shape[0])
        np.random.shuffle(indices)
        # 随机重排每一对变量
        if i >= 2:
            permuted_clinical[:, 2 * i:2 * i + 2] = permuted_clinical[indices][:, 2 * i:2 * i + 2]
        if i==0:
            permuted_clinical[:, 0:4] = permuted_clinical[indices][:, 0:4]
        if i!=1:
            permuted_score = accuracy_score(x, y, permuted_clinical)
            importance = baseline_score - permuted_score
            importances.append(importance)

    return importances

train_x,train_y,train_clinical,test_x,test_y,test_clinical = manage_split(X,Y,clinical)
val_x,val_y,val_clinical = manage_val(val_X,val_Y,val_clinical)
# print('train_x',train_x.shape)
# print('test_x',test_x.shape)
# print('val_x',val_x.shape)
#



net.load_state_dict(torch.load(net_test_pthfile))
v.load_state_dict(torch.load(v_test_pthfile))

# net.load_state_dict(torch.load(net_val_pthfile))
# v.load_state_dict(torch.load(v_val_pthfile))
#


# feature_name = ['age','gender','stone','pain','diabetes','Neutrophil','CA-199','CEA','WBC']
# # 做10次permutation
# data =[]
# n_permute = 10
# for i in range(n_permute):
#     #importances=permutation_importance(train_x,train_y,train_clinical)
#     importances=permutation_importance(test_x,test_y,test_clinical)
#     #importances=permutation_importance(val_x,val_y,val_clinical)
#     data.append(importances)
# print(data)

