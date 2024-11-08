#环境准备
import numpy as np              # numpy数组库
import torch             # torch基础库
import os
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
import re
import pandas as pd

# df = pd.read_csv('/home/hyz/tmp/pycharm_project_519/胆囊癌/训练_clean.csv',encoding='utf-8')
def load_data(CT_data_path,df):
    img_list = []
    label_list = []
    clinical_list = []
    resize = 224

    y_number = 0
    n_number = 0

    path_list = os.listdir(CT_data_path)
    if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
        path_list.remove('.DS_Store')
    if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
        path_list.remove('data_wrong.txt')

    y_path = os.path.join(CT_data_path, 'GBC_merge')
    n_path = os.path.join(CT_data_path, 'Xeno_merge')

    for patient in os.listdir(y_path):
        ind_path = os.path.join(y_path, patient)

        # 找到病人的ID，在Excel表中找到对应的临床数据特征
        patient_id = patient[0:patient.find('_')]
        patient_id = int(patient_id)
        for i in range(df.shape[0]):
            # print(df['编号-数'][i])
            if df['编号-数'][i] == patient_id and df['编号-字母'][i] == 'GBC':

                cli = np.array(eval(df['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list.append(np.array(cli))
                y_number+=1

        # print(ind_path)
        img = Image.open(ind_path)
        img = img.resize((resize, resize))
        img = np.asarray(img)
        img_list.append(np.array(img) / 255.)
        label_list.append(1)

    # print(1)
    for patient in os.listdir(n_path):
        ind_path = os.path.join(n_path, patient)

        patient_id = patient[0:patient.find('_')]
        patient_id = int(patient_id)
        for i in range(df.shape[0]):
            if df['编号-数'][i] == patient_id and df['编号-字母'][i] == 'XGC':
                ## list 变 array

                cli = np.array(eval(df['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list.append(np.array(cli))
                n_number+=1

        # print(ind_path)

        img = Image.open(ind_path)
        img = img.resize((resize, resize))
        img = np.asarray(img)
        img_list.append(np.array(img) / 255.)
        label_list.append(0)

    ### 手动打乱数据集
    clinical = np.array(clinical_list)

    X = np.array(img_list)

    # print(X.shape)

    class_le = LabelEncoder()
    label_encoded = class_le.fit_transform(label_list)

    # print(label_encoded)

    Y = to_categorical(label_encoded, 2)
    # print(Y)  [0,1]代表1

    # print(Y.shape)

    s = np.arange(X.shape[0])
    # print(s)  个数
    np.random.shuffle(s)
    # print(s)

    X = X[s, :, :, :]
    Y = Y[s]
    clinical = clinical[s]


    print(X.shape)
    print(Y.shape)
    print(clinical.shape) # (297, 24)

    print('y_number', y_number)
    print('n_number', n_number)

    return X, Y, clinical

def load_data_val(CT_data_path,df):
    img_list = []
    label_list = []
    clinical_list = []
    resize = 224
    y_number = 0
    n_number =0

    path_list = os.listdir(CT_data_path)
    if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
        path_list.remove('.DS_Store')
    if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
        path_list.remove('data_wrong.txt')

    y_path = os.path.join(CT_data_path, 'GBC_final')
    n_path = os.path.join(CT_data_path, 'xeno_final')

    for patient in os.listdir(y_path):
        ind_path = os.path.join(y_path, patient)

        # 找到病人的ID，在Excel表中找到对应的临床数据特征
        patient_id = patient[0:patient.find('_')]
        patient_id = int(patient_id)
        for i in range(df.shape[0]):
            # print(df['编号-数'][i])
            if df['编号-数'][i] == patient_id and df['编号-字母'][i] == 'GBC':

                cli = np.array(eval(df['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list.append(np.array(cli))
                y_number+=1

        # print(ind_path)
        img = Image.open(ind_path)
        img = img.resize((resize, resize))
        img = np.asarray(img)
        img_list.append(np.array(img) / 255.)
        label_list.append(1)

    # print(1)
    for patient in os.listdir(n_path):
        ind_path = os.path.join(n_path, patient)

        patient_id = patient[0:patient.find('_')]
        patient_id = int(patient_id)
        for i in range(df.shape[0]):
            if df['编号-数'][i] == patient_id and df['编号-字母'][i] == 'XGC':
                ## list 变 array

                cli = np.array(eval(df['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list.append(np.array(cli))
                n_number += 1

        # print(ind_path)

        img = Image.open(ind_path)
        img = img.resize((resize, resize))
        img = np.asarray(img)
        img_list.append(np.array(img) / 255.)
        label_list.append(0)

    ### 手动打乱数据集
    clinical = np.array(clinical_list)

    X = np.array(img_list)

    # print(X.shape)

    class_le = LabelEncoder()
    label_encoded = class_le.fit_transform(label_list)

    # print(label_encoded)

    Y = to_categorical(label_encoded, 2)
    # print(Y)  [0,1]代表1

    # print(Y.shape)

    s = np.arange(X.shape[0])
    # print(s)  个数
    np.random.shuffle(s)
    # print(s)

    X = X[s, :, :, :]
    Y = Y[s]
    clinical = clinical[s]


    print(X.shape)
    print(Y.shape)
    print(clinical.shape)

    print('y_number', y_number)
    print('n_number', n_number)



    return X, Y, clinical

# def load_data_imageonly(CT_data_path):
#     img_list = []
#     label_list = []
#
#     resize = 224
#
#     path_list = os.listdir(CT_data_path)
#     if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
#         path_list.remove('.DS_Store')
#     if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
#         path_list.remove('data_wrong.txt')
#
#     y_path = os.path.join(CT_data_path, 'GBC_merge')
#     n_path = os.path.join(CT_data_path, 'Xeno_merge')
#
#     for patient in os.listdir(y_path):
#         ind_path = os.path.join(y_path, patient)
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(1)
#
#     for patient in os.listdir(n_path):
#         ind_path = os.path.join(n_path, patient)
#
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(0)
#
#
#     X = np.array(img_list)
#
#
#     class_le = LabelEncoder()
#     label_encoded = class_le.fit_transform(label_list)
#     Y = to_categorical(label_encoded, 2)
#     s = np.arange(X.shape[0])
#     np.random.shuffle(s)
#     X = X[s, :, :, :]
#     Y = Y[s]
#
#     print(X.shape)
#     print(Y.shape)
#     return X, Y
#
# def load_data_imageonly_val(CT_data_path):
#     img_list = []
#     label_list = []
#     resize = 224
#
#     path_list = os.listdir(CT_data_path)
#     if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
#         path_list.remove('.DS_Store')
#     if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
#         path_list.remove('data_wrong.txt')
#
#     y_path = os.path.join(CT_data_path, 'GBC_final')
#     n_path = os.path.join(CT_data_path, 'xeno_final')
#
#     for patient in os.listdir(y_path):
#         ind_path = os.path.join(y_path, patient)
#
#
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(1)
#
#     # print(1)
#     for patient in os.listdir(n_path):
#         ind_path = os.path.join(n_path, patient)
#
#
#
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(0)
#
#
#     X = np.array(img_list)
#     class_le = LabelEncoder()
#     label_encoded = class_le.fit_transform(label_list)
#
#     Y = to_categorical(label_encoded, 2)
#
#     s = np.arange(X.shape[0])
#     # print(s)  个数
#     np.random.shuffle(s)
#     # print(s)
#
#     X = X[s, :, :, :]
#     Y = Y[s]
#
#
#
#     print(X.shape)
#     print(Y.shape)
#
#
#     return X, Y
#
#
# def load_data_imageonly_new(CT_data_path):
#     img_list = []
#     label_list = []
#
#     resize = 224
#
#     path_list = os.listdir(CT_data_path)
#     if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
#         path_list.remove('.DS_Store')
#     if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
#         path_list.remove('data_wrong.txt')
#
#     y_path = os.path.join(CT_data_path, 'gbc','merge')
#     n_path = os.path.join(CT_data_path, 'xgc','merge')
#
#     for patient in os.listdir(y_path):
#         ind_path = os.path.join(y_path, patient)
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(1)
#
#     for patient in os.listdir(n_path):
#         ind_path = os.path.join(n_path, patient)
#
#         img = Image.open(ind_path)
#         img = img.resize((resize, resize))
#         img = np.asarray(img)
#         img_list.append(np.array(img) / 255.)
#         label_list.append(0)
#
#     X = np.array(img_list)
#
#     class_le = LabelEncoder()
#     label_encoded = class_le.fit_transform(label_list)
#     Y = to_categorical(label_encoded, 2)
#     s = np.arange(X.shape[0])
#     np.random.shuffle(s)
#     X = X[s, :, :, :]
#     Y = Y[s]
#
#     print(X.shape)
#     print(Y.shape)
#     return X, Y

def load_data_new(CT_data_path):
    ## 庆春
    img_list1 = []
    label_list1 = []
    clinical_list1 = []

    ###余杭和之江

    img_list2 = []
    label_list2 = []
    clinical_list2 = []
    resize = 224

    path_list = os.listdir(CT_data_path)
    if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
        path_list.remove('.DS_Store')
    if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
        path_list.remove('data_wrong.txt')

    y_path = os.path.join(CT_data_path, 'gbc', 'merge')
    n_path = os.path.join(CT_data_path, 'xgc', 'merge')

    df_gbc = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/最新胆囊数据/胆囊癌_clean.csv')
    df_xgc = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/最新胆囊数据/胆囊黄色肉芽肿_clean.csv')
    number_y = 0
    number_n = 0
    nannangai_list = []
    rouya_list = []
    for patient in os.listdir(y_path):
        ind_path = os.path.join(y_path, patient)
        number_y+=1

        # 找到病人的ID，在Excel表中找到对应的临床数据特征
        patient_id = re.search(r'\d+', patient).group()
        patient_id = int(patient_id)
        print('胆囊癌id', patient_id)
        nannangai_list.append(patient_id)

        for i in range(df_gbc.shape[0]):
            # print(df['编号-数'][i])
            if df_gbc['序号'][i] == patient_id and int(patient_id)<=14:

                cli = np.array(eval(df_gbc['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list1.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list1.append(np.array(img) / 255.)
                label_list1.append(1)

            if df_gbc['序号'][i] == patient_id and int(patient_id)>14:

                cli = np.array(eval(df_gbc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list2.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list2.append(np.array(img) / 255.)
                label_list2.append(1)


    # print(1)
    for patient in os.listdir(n_path):
        ind_path = os.path.join(n_path, patient)
        number_n +=1

        patient_id = re.search(r'\d+', patient).group()
        patient_id = int(patient_id)
        print('黄色肉芽肿id', patient_id)
        rouya_list.append(patient_id)

        for i in range(df_xgc.shape[0]):
            # print(df['编号-数'][i])
            if df_xgc['序号'][i] == patient_id  and int(patient_id)<=60:
                #print('序号：',df_xgc['序号'][i])
                cli = np.array(eval(df_xgc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list1.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list1.append(np.array(img) / 255.)
                label_list1.append(0)

            if df_xgc['序号'][i] == patient_id  and int(patient_id)>60:
                #print('序号：',df_xgc['序号'][i])
                cli = np.array(eval(df_xgc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list2.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list2.append(np.array(img) / 255.)
                label_list2.append(0)

    print('number_y',number_y)
    print('number_n', number_n)
    print('胆囊癌id')
    for i in sorted(nannangai_list):
        print(i)
    print('黄色肉')
    for i in sorted(rouya_list):
        print(i)
    ### 手动打乱数据集
    clinical1 = np.array(clinical_list1)
    clinical2 = np.array(clinical_list2)


    X1 = np.array(img_list1)
    X2 = np.array(img_list2)


    # print(X.shape)

    class_le = LabelEncoder()
    label_encoded1 = class_le.fit_transform(label_list1)
    label_encoded2 = class_le.fit_transform(label_list2)

    # print(label_encoded)

    Y1 = to_categorical(label_encoded1, 2)
    Y2 = to_categorical(label_encoded2, 2)


    s1 = np.arange(X1.shape[0])
    s2 = np.arange(X2.shape[0])

    # print(s)  个数
    np.random.shuffle(s1)
    np.random.shuffle(s2)
    # print(s)

    X1 = X1[s1, :, :, :]
    X2 = X2[s2, :, :, :]

    Y1 = Y1[s1]
    Y2 = Y2[s2]

    clinical1 = clinical1[s1]
    clinical2 = clinical2[s2]


    print(X1.shape)
    print(Y1.shape)
    print(clinical1.shape)

    print(X2.shape)
    print(Y2.shape)
    print(clinical2.shape)

    return X1, Y1, clinical1,X2, Y2, clinical2

def load_data_3(CT_data_path):
    ## 庆春
    img_list1 = []
    label_list1 = []
    clinical_list1 = []

    ###余杭和之江

    img_list2 = []
    label_list2 = []
    clinical_list2 = []
    resize = 224

    path_list = os.listdir(CT_data_path)
    if os.path.exists(os.path.join(CT_data_path, '.DS_Store')):
        path_list.remove('.DS_Store')
    if os.path.exists(os.path.join(CT_data_path, 'data_wrong.txt')):
        path_list.remove('data_wrong.txt')

    y_path = os.path.join(CT_data_path, 'gbc', 'merge')
    n_path = os.path.join(CT_data_path, 'xgc', 'merge')

    df_gbc = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/胆囊数据第三次/胆囊癌_list.csv')
    df_xgc = pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/胆囊数据第三次/黄色肉芽肿_list.csv')
    number_y = 0
    number_n = 0
    nannangai_list = []
    rouya_list = []
    for patient in os.listdir(y_path):
        ind_path = os.path.join(y_path, patient)
        number_y+=1

        # 找到病人的ID，在Excel表中找到对应的临床数据特征
        patient_id = re.search(r'\d+', patient).group()
        patient_id = int(patient_id)
        print('胆囊癌id', patient_id)
        nannangai_list.append(patient_id)

        for i in range(df_gbc.shape[0]):
            # print(df['编号-数'][i])
            if df_gbc['序号'][i] == patient_id and int(patient_id)<=21:

                cli = np.array(eval(df_gbc['list'][i]))
                #cli = cli.reshape([1, len(cli)])
                clinical_list1.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list1.append(np.array(img) / 255.)
                label_list1.append(1)

            if df_gbc['序号'][i] == patient_id and int(patient_id)>21:

                cli = np.array(eval(df_gbc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list2.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list2.append(np.array(img) / 255.)
                label_list2.append(1)


    # print(1)
    for patient in os.listdir(n_path):
        ind_path = os.path.join(n_path, patient)
        number_n +=1

        patient_id = re.search(r'\d+', patient).group()
        patient_id = int(patient_id)
        print('黄色肉芽肿id', patient_id)
        rouya_list.append(patient_id)

        for i in range(df_xgc.shape[0]):
            # print(df['编号-数'][i])
            if df_xgc['序号'][i] == patient_id  and int(patient_id)<=25:
                #print('序号：',df_xgc['序号'][i])
                cli = np.array(eval(df_xgc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list1.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list1.append(np.array(img) / 255.)
                label_list1.append(0)

            if df_xgc['序号'][i] == patient_id  and int(patient_id)>25:
                #print('序号：',df_xgc['序号'][i])
                cli = np.array(eval(df_xgc['list'][i]))
                # cli = cli.reshape([1, len(cli)])
                clinical_list2.append(np.array(cli))

                img = Image.open(ind_path)
                img = img.resize((resize, resize))
                img = np.asarray(img)
                img_list2.append(np.array(img) / 255.)
                label_list2.append(0)

    # print('number_y',number_y)
    # print('number_n', number_n)
    # print('胆囊癌id')
    # for i in sorted(nannangai_list):
    #     print(i)
    # print('黄色肉')
    # for i in sorted(rouya_list):
    #     print(i)
    ### 手动打乱数据集
    clinical1 = np.array(clinical_list1)
    clinical2 = np.array(clinical_list2)


    X1 = np.array(img_list1)
    X2 = np.array(img_list2)


    # print(X.shape)

    class_le = LabelEncoder()
    label_encoded1 = class_le.fit_transform(label_list1)
    label_encoded2 = class_le.fit_transform(label_list2)

    # print(label_encoded)

    Y1 = to_categorical(label_encoded1, 2)
    Y2 = to_categorical(label_encoded2, 2)


    s1 = np.arange(X1.shape[0])
    s2 = np.arange(X2.shape[0])

    # print(s)  个数
    np.random.shuffle(s1)
    np.random.shuffle(s2)
    # print(s)

    X1 = X1[s1, :, :, :]
    X2 = X2[s2, :, :, :]

    Y1 = Y1[s1]
    Y2 = Y2[s2]

    clinical1 = clinical1[s1]
    clinical2 = clinical2[s2]



    print('X1.shape',X1.shape)
    print('Y1.shape',Y1.shape)
    print('clinical1.shape',clinical1.shape)

    print(X2.shape)
    print(Y2.shape)
    print(clinical2.shape)

    return X1, Y1, clinical1,X2, Y2, clinical2

X, Y, clinical = load_data_val('/home/hpc/users/wangqing/gallbladder/gallbladder/val/merge_cancer',pd.read_csv('/home/hpc/users/hanyuzhe/胆囊_vit/验证_clean.csv'))
# label_test = np.load('/home/hpc/users/hanyuzhe/胆囊_vit/np_imageonly/test_label.npy')
# print(label_test.shape)