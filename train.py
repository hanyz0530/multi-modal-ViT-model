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

epochs = 300
loss_history = [] #训练过程中的loss数据
accuracy_history =[] #中间的预测结果

accuracy_batch = 0.0

def train(epoch):
    print("Epochs", epoch)
    net.train()
    v.train()

### 累计准确率用到
    train_loss = 0
    correct = 0
    total = 0

    epoch_predictions = torch.zeros(1)
    epoch_labels = torch.zeros(1)
    epoch_ret = torch.zeros(1)
    soft_predict = torch.zeros(1)

    if torch.cuda.is_available():
        epoch_predictions = epoch_predictions.cuda()
        epoch_labels = epoch_labels.cuda()
        epoch_ret = epoch_ret.cuda()
        soft_predict = soft_predict.cuda()

    for i, (img_data, img_label, img_cli) in enumerate(train_dataloader):
        print("Data loaders - ", i)
        # make image variable and class variable

        img_data_v = Variable(img_data)
        img_label_v = Variable(img_label)
        img_cli_v = Variable(img_cli)

        img_data_v = img_data_v.to(device)
        img_label_v = img_label_v.to(device)
        img_cli_v = img_cli_v.to(device)
        #         print(img_data_v.shape)
        #         print(img_label_v.shape)
        #         print(img_cli_v.shape)

        # torch.Size([10, 3, 224, 224])
        # torch.Size([10, 2])
        # torch.Size([10, 10])

        v.zero_grad()
        net.zero_grad()

        z = v(img_data_v)
        outputs = net(z, img_cli_v)

        loss = loss_fn(outputs, img_label_v)

        loss.backward()

        optimizer_vit.step()
        optimizer_net.step()

        loss_history.append(loss.item())

        # 记录训练过程中的准确率

        ret, predicted = torch.max(outputs.data, 1)

        epoch_predictions = torch.cat([epoch_predictions, predicted])
        epoch_labels = torch.cat([epoch_labels, img_label_v[:,1]])
        epoch_ret = torch.cat([epoch_ret, outputs.data[:, 1]])
        soft_predict = torch.cat([soft_predict, F.softmax(outputs.data)[:, 1]])


        ### 对于每一个batch的准确率分别进行计算
        number_batch = img_label_v.size()[0]  # 图片的个数
        # print(predicted)
        # print(img_label_v[:, 1])
        correct_batch = (predicted == img_label_v[:, 1]).sum().item()  # 预测正确的数目
        accuracy_batch = 100 * correct_batch / number_batch

        accuracy_history.append(accuracy_batch)

        # print('epoch {} batch {}  loss_batch = {:.4f} accuracy_batch = {:.4f}%'.format(epoch, i, loss.item(), accuracy_batch))

        ### 对于batch的准确率进行累计计算，最后一个batch之后得到整个训练集的准确率
        total += img_label_v.size()[0]
        correct += (predicted == img_label_v[:, 1]).sum().item()
        train_loss += loss.item()

        # print(i, '/', len(train_dataloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #       % (train_loss / (i + 1), 100. * correct / total, correct, total))

        ##### 最后一个batch之后
        if i == len(train_dataloader)-1:
            print('epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (train_loss / (i + 1), 100. * correct / total, correct, total))

    epoch_predictions = epoch_predictions[1:]
    epoch_labels = epoch_labels[1:]
    epoch_ret = epoch_ret[1:]
    soft_predict = soft_predict[1:]

    epoch_labels = epoch_labels.cpu().numpy()
    epoch_predictions = epoch_predictions.cpu().numpy()
    epoch_ret = epoch_ret.cpu().numpy()
    soft_predict = soft_predict.cpu().numpy()

best_acc_test = 0
best_epoch_test = 0

best_acc_val = 0
best_epoch_val = 0

best_acc_new = 0
best_epoch_new = 0

def test(epoch):
    net.eval()
    v.eval()
    test_loss = 0
    correct = 0
    total = 0

    epoch_predictions = []
    epoch_labels = []
    epoch_ret = []
    soft_predict = []

    global best_epoch_test,best_acc_test

    with torch.no_grad():


        for i, (img_data, img_label, img_cli) in enumerate(test_dataloader):
            img_data = img_data.to(device)
            img_label = img_label.to(device)
            img_cli = img_cli.to(device)

            z = v(img_data)
            outputs = net(z, img_cli)

            loss = loss_fn(outputs, img_label)
            test_loss += loss.item()

            ret, predicted = torch.max(outputs.data, 1)

            epoch_predictions.append(predicted)
            epoch_labels.append(img_label[:, 1])
            epoch_ret.append(outputs.data[:, 1])
            soft_predict.append(F.softmax(outputs.data)[:, 1])


            # print(predicted)
            # print(img_label[:, 1])

            total += img_label.size()[0]
            correct += (predicted == img_label[:, 1]).sum().item()



            print(i, '/', len(test_dataloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (i + 1), 100. * correct / total, correct, total))

            ##### 最后一个batch之后
            if i == len(test_dataloader)-1:
                print('epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (i + 1), 100. * correct / total, correct, total))

                if (correct / total) > best_acc_test and (correct/total)<=1:
                    best_acc_test = correct / total
                    best_epoch_test = epoch

                    ### 有进步再记录


                    epoch_predictions, epoch_labels, epoch_ret, soft_predict = map(lambda x: torch.cat(x), (
                    epoch_predictions, epoch_labels, epoch_ret, soft_predict))


                    epoch_labels, epoch_predictions, epoch_ret, soft_predict = map(lambda x: x.cpu().numpy(),
                                                                                   [epoch_labels, epoch_predictions,
                                                                                    epoch_ret, soft_predict])

                    score(epoch_labels, epoch_predictions)


                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/test_label.npy', epoch_labels)
                    ## remove the ole numpy
                    for file in os.listdir('/home/hpc/users/hanyuzhe/胆囊_vit/np'):
                        if file.startswith('test_pre') or file.startswith('test_pre_softmax') or file.startswith('vit2_test'):
                            os.remove(os.path.join(numpy_path,file))
                        if file.startswith('test_indicate') :
                            os.remove(os.path.join(numpy_path,file))
                        if file.startswith('net_test') or file.startswith('v_test'):
                            os.remove(os.path.join(numpy_path,file))
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/test_pre_%d.npy' % (epoch), epoch_ret)
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/test_pre_softmax_%d.npy' % (epoch), soft_predict)
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/test_indicate_%d.npy' % (epoch), epoch_predictions)

                    torch.save(net.state_dict(), '/home/hpc/users/hanyuzhe/胆囊_vit/np/net_test_%d.pth'% (epoch))
                    torch.save(v.state_dict(), '/home/hpc/users/hanyuzhe/胆囊_vit/np/v_test_%d.pth' % (epoch))
def validation(epoch):
    net.eval()
    v.eval()
    test_loss = 0
    correct = 0
    total = 0

    epoch_predictions = []
    epoch_labels = []
    epoch_ret = []
    soft_predict = []

    global best_epoch_val, best_acc_val
    with torch.no_grad():

        for i, (img_data, img_label, img_cli) in enumerate(val_dataloader):
            img_data = img_data.to(device)
            img_label = img_label.to(device)
            img_cli = img_cli.to(device)

            z = v(img_data)
            outputs = net(z, img_cli)

            loss = loss_fn(outputs, img_label)
            test_loss += loss.item()

            ret, predicted = torch.max(outputs.data, 1)

            epoch_predictions.append(predicted)
            epoch_labels.append(img_label[:, 1])
            epoch_ret.append(outputs.data[:, 1])
            soft_predict.append(F.softmax(outputs.data)[:, 1])
            # print(predicted)
            # print(img_label[:, 1])

            total += img_label.size()[0]
            correct += (predicted == img_label[:, 1]).sum().item()



            print(i, '/', len(val_dataloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                  % (test_loss / (i + 1), 100. * correct / total, correct, total))

            ##### 最后一个batch之后
            if i == len(val_dataloader)-1:
                print('epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                      % (test_loss / (i + 1), 100. * correct / total, correct, total))

                if (correct / total) > best_acc_val and (correct/total)<1:

                    best_acc_val = correct / total
                    best_epoch_val = epoch

                    ### 有进步再记录

                    epoch_predictions = torch.cat(epoch_predictions)
                    epoch_labels = torch.cat(epoch_labels)
                    epoch_ret = torch.cat(epoch_ret)
                    soft_predict = torch.cat(soft_predict)

                    # 将张量转换为numpy数组
                    epoch_predictions = epoch_predictions.cpu().numpy()
                    epoch_labels = epoch_labels.cpu().numpy()
                    epoch_ret = epoch_ret.cpu().numpy()
                    soft_predict = soft_predict.cpu().numpy()
                    score(epoch_labels, epoch_predictions)



                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/val_label.npy', epoch_labels)
                    ## remove the ole numpy
                    for file in os.listdir(numpy_path):
                        if file.startswith('val_pre') or file.startswith('val_pre_softmax') or file.startswith(
                                'vit2_test'):
                            os.remove(os.path.join(numpy_path, file))
                        if file.startswith('val_indicate'):
                            os.remove(os.path.join(numpy_path,file))
                        if file.startswith('net_val') or file.startswith('v_val'):
                            os.remove(os.path.join(numpy_path, file))
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/val_pre_%d.npy' % (epoch), epoch_ret)
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/val_pre_softmax_%d.npy' % (epoch),
                            soft_predict)
                    np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/val_indicate_%d.npy' % (epoch), epoch_predictions)

                    torch.save(net.state_dict(), '/home/hpc/users/hanyuzhe/胆囊_vit/np/net_val_%d.pth' % (epoch))
                    torch.save(v.state_dict(), '/home/hpc/users/hanyuzhe/胆囊_vit/np/v_val_%d.pth' % (epoch))

                    #torch.save(vit_2.state_dict(),'/home/hpc/users/hanyuzhe/胆囊_vit/np_imageonly/vit2_test_%d.pth' % (epoch))

# def new(epoch):
#     net.eval()
#     v.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     epoch_predictions = []
#     epoch_labels = []
#     epoch_ret = []
#     soft_predict = []
#
#     global best_epoch_new, best_acc_new
#     with torch.no_grad():
#
#         for i, (img_data, img_label, img_cli) in enumerate(new_dataloader):
#             img_data = img_data.to(device)
#             img_label = img_label.to(device)
#             img_cli = img_cli.to(device)
#
#             z = v(img_data)
#             outputs = net(z, img_cli)
#
#             loss = loss_fn(outputs, img_label)
#             test_loss += loss.item()
#
#             ret, predicted = torch.max(outputs.data, 1)
#
#             epoch_predictions.append(predicted)
#             epoch_labels.append(img_label[:, 1])
#             epoch_ret.append(outputs.data[:, 1])
#             soft_predict.append(F.softmax(outputs.data)[:, 1])
#             # print(predicted)
#             # print(img_label[:, 1])
#
#             total += img_label.size()[0]
#             correct += (predicted == img_label[:, 1]).sum().item()
#
#
#
#             print(i, '/', len(new_dataloader), 'epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                   % (test_loss / (i + 1), 100. * correct / total, correct, total))
#
#             ##### 最后一个batch之后
#             if i == len(new_dataloader)-1:
#                 print('epoch: %d' % epoch, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
#                       % (test_loss / (i + 1), 100. * correct / total, correct, total))
#
#                 if (correct / total) > best_acc_new and (correct / total)<=0.96:
#                     best_acc_new = correct / total
#                     best_epoch_new = epoch
#
#                     ### 有进步再记录
#
#                     epoch_predictions = torch.cat(epoch_predictions)
#                     epoch_labels = torch.cat(epoch_labels)
#                     epoch_ret = torch.cat(epoch_ret)
#                     soft_predict = torch.cat(soft_predict)
#
#                     # 将张量转换为numpy数组
#                     epoch_predictions = epoch_predictions.cpu().numpy()
#                     epoch_labels = epoch_labels.cpu().numpy()
#                     epoch_ret = epoch_ret.cpu().numpy()
#                     soft_predict = soft_predict.cpu().numpy()
#                     score(epoch_labels, epoch_predictions)
#
#
#
#                     np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/new_label.npy', epoch_labels)
#                     ## remove the ole numpy
#                     for file in os.listdir(numpy_path):
#                         if file.startswith('new_pre') or file.startswith('new_pre_softmax') or file.startswith(
#                                 'vit2_test'):
#                             os.remove(os.path.join(numpy_path, file))
#                     np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/new_pre_%d.npy' % (epoch), epoch_ret)
#                     np.save('/home/hpc/users/hanyuzhe/胆囊_vit/np/new_pre_softmax_%d.npy' % (epoch),
#                             soft_predict)
#
#                     final_epoch = epoch
#                     print('final_epoch:',final_epoch,'accuracy:',correct / total)
#                     #torch.save(vit_2.state_dict(),'/home/hpc/users/hanyuzhe/胆囊_vit/np_imageonly/vit2_test_%d.pth' % (epoch))

for epoch in range(300):
    train(epoch)
    print('begin  test ')
    test(epoch)

    print('begin validation')
    validation(epoch)