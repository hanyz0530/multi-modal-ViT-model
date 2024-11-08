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

### 删除文件
for filename in os.listdir(numpy_path):
    file_path = os.path.join(numpy_path, filename) # 拼接文件路径
    if os.path.isfile(file_path): # 如果是文件则删除
        os.remove(file_path)

def score(epoch_labels, best_predictions):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    for k in range(len(epoch_labels)):
        i = epoch_labels[k]
        j = best_predictions[k]
        if i == j and i == 1:
            tp += 1
        if i == j and i == 0:
            tn += 1
        if i != j and i == 1:
            fn += 1
        if i != j and i == 0:
            fp += 1
    # print(tp, tn, fp, fn)
    if tp + fp > 0 and tp + fn > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        if precision + recall > 0:
            F1 = 2 * precision * recall / (precision + recall)
            print("precision: %f  recall: %f   F1_score: %f" % (precision, recall, F1))

v = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
# Create a new Linear layer with 2 output features
new_head = torch.nn.Linear(in_features=192, out_features=1024, bias=True)
# Replace the head layer of the model with the new Linear layer
v.head = new_head

# v = ViT(
#     image_size = 256,
#     patch_size = 32,
#     num_classes = 1024,
#     dim = 1024,
#     depth = 6,
#     heads = 16,
#     mlp_dim = 2048,
#     dropout = 0.1,
#     emb_dropout = 0.1
# )

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