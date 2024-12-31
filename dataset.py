import logging
from collections import OrderedDict
import pandas as pd
import flwr as fl
import numpy as np
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import requests







def load_data5(partition_id):
    # """Load partition data."""

    logging.info("Loading data from Excel files...")
    df = pd.read_excel('dataset/tsu.xlsx')
    rows = df.shape[0]

    # 选择特征列和标签列
    features = ['mid-term', 'faculty', 'department']
    label = 'Label'

    # 数据清洗，例如删除缺失值
    df = df.dropna()

    # 特征编码和缩放
    numerical_cols = ['mid-term']
    categorical_cols = ['faculty', 'department']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    # 划分训练集和测试集
    X = df[features]
    y = df[label].values
    X1 = X[:309]
    X2 = X[309:889]
    X3 = X[889:1293]
    X4 = X[1293:1505]
    X5 = X[1293:]
    y1 = y[:309]
    y2 = y[309:889]
    y3 = y[889:1293]
    y4 = y[1293:1505]
    y5 = y[1293:]

    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=42)
    X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.2, random_state=42)
    X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.2, random_state=42)


    preprocessor.fit(df[features])
    X1_train_encoded = preprocessor.transform(X1_train)
    X1_test_encoded = preprocessor.transform(X1_test)
    X2_train_encoded = preprocessor.transform(X2_train)
    X2_test_encoded = preprocessor.transform(X2_test)
    X3_train_encoded = preprocessor.transform(X3_train)
    X3_test_encoded = preprocessor.transform(X3_test)
    X4_train_encoded = preprocessor.transform(X4_train)
    X4_test_encoded = preprocessor.transform(X4_test)
    X5_train_encoded = preprocessor.transform(X5_train)
    X5_test_encoded = preprocessor.transform(X5_test)
    # 假设有5个分区的数据集
    datasets_train = {
        'X1_train_encode': X1_train_encoded,
        'X2_train_encode': X2_train_encoded,
        'X3_train_encode': X3_train_encoded,
        'X4_train_encode': X4_train_encoded,
        'X5_train_encode': X5_train_encoded
    }
    datasets_test = {
        'X1_test_encode': X1_test_encoded,
        'X2_test_encode': X2_test_encoded,
        'X3_test_encode': X3_test_encoded,
        'X4_test_encode': X4_test_encoded,
        'X5_test_encode': X5_test_encoded
    }
    # 在ColumnTransformer预处理之后检查转换后的特征数



    X_train_encoded = datasets_train[f'X{partition_id}_train_encode']
    X_test_encoded = datasets_test[f'X{partition_id}_test_encode']
    print("Number of features after transformation:", X_train_encoded.shape[1])

    # 将编码后的数据转换为PyTorch张量
    # 假设 X_train_encoded 是一个稀疏矩阵
    # 首先，将其转换为密集矩阵
    X_train_encoded_dense = X_train_encoded.toarray()
    # 然后，使用密集矩阵创建 PyTorch 张量
    X_train_tensor = torch.tensor(X_train_encoded_dense, dtype=torch.float32)
    # 重复相同的步骤来处理测试集数据
    X_test_encoded_dense = X_test_encoded.toarray()
    X_test_tensor = torch.tensor(X_test_encoded_dense, dtype=torch.float32)
    y1_train_tensor = torch.tensor(y1_train, dtype=torch.long)
    y1_test_tensor = torch.tensor(y1_test, dtype=torch.long)
    y2_train_tensor = torch.tensor(y2_train, dtype=torch.long)
    y2_test_tensor = torch.tensor(y2_test, dtype=torch.long)
    y3_train_tensor = torch.tensor(y3_train, dtype=torch.long)
    y3_test_tensor = torch.tensor(y3_test, dtype=torch.long)
    y4_train_tensor = torch.tensor(y4_train, dtype=torch.long)
    y4_test_tensor = torch.tensor(y4_test, dtype=torch.long)
    y5_train_tensor = torch.tensor(y5_train, dtype=torch.long)
    y5_test_tensor = torch.tensor(y5_test, dtype=torch.long)
    data_train = {
        'y1_train_tensor': y1_train_tensor,
        'y2_train_tensor': y2_train_tensor,
        'y3_train_tensor': y3_train_tensor,
        'y4_train_tensor': y4_train_tensor,
        'y5_train_tensor': y5_train_tensor
    }
    data_test = {
        'y1_test_tensor': y1_test_tensor,
        'y2_test_tensor': y2_test_tensor,
        'y3_test_tensor': y3_test_tensor,
        'y4_test_tensor': y4_test_tensor,
        'y5_test_tensor': y5_test_tensor
    }
    y_train_tensor = data_train[f'y{partition_id}_train_tensor']
    y_test_tensor = data_test[f'y{partition_id}_test_tensor']


    X_test_tensor1 = X_test_tensor.reshape(-1, 1, X_test_tensor.shape[1], 1)
    X_train_tensor1 = X_train_tensor.reshape(-1, 1, X_train_tensor.shape[1], 1)

    # 创建TensorDataset和DataLoader

    train_dataset = TensorDataset(X_train_tensor1, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor1, y_test_tensor)




    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False)

    return train_loader, test_loader


# 特征选择，不带期中
def load_data2choose(partition_id):
    # """Load partition data."""

    logging.info("Loading data from Excel files...")

    df = pd.read_excel('dataset/sau1.xlsx')

    rows = df.shape[0]

    # 选择特征列和标签列

    features = ['phy1','math2', 'circuit','pro', 'sex', 'zyy']

    label = 'Label'

    # 数据清洗，例如删除缺失值
    df = df.dropna()

    # 特征编码和缩放

    numerical_cols = ['phy1','math2', 'circuit','pro','sex']
    categorical_cols = ['sex', 'zyy']


    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    # 划分训练集和测试集
    X = df[features]
    y = df[label].values
    X1 = X[:859]
    X2 = X[859:]
    y1 = y[:859]
    y2 = y[859:]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
    # print("fengehou",len(X1),len(X2))
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessor.fit(df[features])
    X1_train_encoded = preprocessor.transform(X1_train)
    X1_test_encoded = preprocessor.transform(X1_test)
    X2_train_encoded = preprocessor.transform(X2_train)
    X2_test_encoded = preprocessor.transform(X2_test)
    # print(X_train_encoded.shape)
    # print(X_test_encoded.shape)

    # 将编码后的数据转换为PyTorch张量
    # 假设 X_train_encoded 是一个稀疏矩阵
    # 首先，将其转换为密集矩阵
    # X1_train_encoded_dense = X1_train_encoded.toarray()
    # X2_train_encoded_dense = X2_train_encoded.toarray()
    X1_train_encoded_dense = X1_train_encoded
    X2_train_encoded_dense = X2_train_encoded
    # 然后，使用密集矩阵创建 PyTorch 张量
    X1_train_tensor = torch.tensor(X1_train_encoded_dense, dtype=torch.float32)
    X2_train_tensor = torch.tensor(X2_train_encoded_dense, dtype=torch.float32)
    # 重复相同的步骤来处理测试集数据
    # X1_test_encoded_dense = X1_test_encoded.toarray()
    # X2_test_encoded_dense = X2_test_encoded.toarray()
    X1_test_encoded_dense = X1_test_encoded
    X2_test_encoded_dense = X2_test_encoded
    X1_test_tensor = torch.tensor(X1_test_encoded_dense, dtype=torch.float32)
    X2_test_tensor = torch.tensor(X2_test_encoded_dense, dtype=torch.float32)
    y1_train_tensor = torch.tensor(y1_train, dtype=torch.long)
    y1_test_tensor = torch.tensor(y1_test, dtype=torch.long)
    y2_train_tensor = torch.tensor(y2_train, dtype=torch.long)
    y2_test_tensor = torch.tensor(y2_test, dtype=torch.long)

    X1_test_tensor1 = X1_test_tensor.reshape(-1, 1, X1_test_tensor.shape[1], 1)
    X1_train_tensor1 = X1_train_tensor.reshape(-1, 1, X1_train_tensor.shape[1], 1)
    X2_test_tensor1 = X2_test_tensor.reshape(-1, 1, X2_test_tensor.shape[1], 1)
    X2_train_tensor1 = X2_train_tensor.reshape(-1, 1, X2_train_tensor.shape[1], 1)

    # 创建TensorDataset和DataLoader

    train1_dataset = TensorDataset(X1_train_tensor1, y1_train_tensor)
    test1_dataset = TensorDataset(X1_test_tensor1, y1_test_tensor)
    train2_dataset = TensorDataset(X2_train_tensor1, y2_train_tensor)
    test2_dataset = TensorDataset(X2_test_tensor1, y2_test_tensor)

    if partition_id == 1:
        train_loader = DataLoader(dataset=train1_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=test1_dataset, batch_size=256, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train2_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=test2_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader
#特征选择，带期中的
def load_data2(partition_id):
    # """Load partition data."""

    logging.info("Loading data from Excel files...")

    df = pd.read_excel('dataset/sau2.xlsx')

    rows = df.shape[0]

    # 选择特征列和标签列


    features= ['phy1', 'math2', 'circuit', 'pro','cj','sex', 'zyy']

    label = 'Label'

    # 数据清洗，例如删除缺失值
    df = df.dropna()

    # 特征编码和缩放
    numerical_cols = ['phy1', 'math2', 'circuit', 'pro','cj']
    categorical_cols = ['sex','zyy']



    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ]
    )
    # 划分训练集和测试集
    X = df[features]
    y = df[label].values

    X1 = X[:795]
    X2 = X[795:]
    y1 = y[:795]
    y2 = y[795:]
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

    preprocessor.fit(df[features])
    X1_train_encoded = preprocessor.transform(X1_train)
    X1_test_encoded = preprocessor.transform(X1_test)
    X2_train_encoded = preprocessor.transform(X2_train)
    X2_test_encoded = preprocessor.transform(X2_test)
    # print(X_train_encoded.shape)
    # print(X_test_encoded.shape)

    # 将编码后的数据转换为PyTorch张量
    # 假设 X_train_encoded 是一个稀疏矩阵
    # 首先，将其转换为密集矩阵
    # X1_train_encoded_dense = X1_train_encoded.toarray()
    # X2_train_encoded_dense = X2_train_encoded.toarray()
    X1_train_encoded_dense = X1_train_encoded
    X2_train_encoded_dense = X2_train_encoded
    # 然后，使用密集矩阵创建 PyTorch 张量
    X1_train_tensor = torch.tensor(X1_train_encoded_dense, dtype=torch.float32)
    X2_train_tensor = torch.tensor(X2_train_encoded_dense, dtype=torch.float32)
    # 重复相同的步骤来处理测试集数据
    # X1_test_encoded_dense = X1_test_encoded.toarray()
    # X2_test_encoded_dense = X2_test_encoded.toarray()
    X1_test_encoded_dense = X1_test_encoded
    X2_test_encoded_dense = X2_test_encoded
    X1_test_tensor = torch.tensor(X1_test_encoded_dense, dtype=torch.float32)
    X2_test_tensor = torch.tensor(X2_test_encoded_dense, dtype=torch.float32)
    y1_train_tensor = torch.tensor(y1_train, dtype=torch.long)
    y1_test_tensor = torch.tensor(y1_test, dtype=torch.long)
    y2_train_tensor = torch.tensor(y2_train, dtype=torch.long)
    y2_test_tensor = torch.tensor(y2_test, dtype=torch.long)

    X1_test_tensor1 = X1_test_tensor.reshape(-1, 1, X1_test_tensor.shape[1], 1)
    X1_train_tensor1 = X1_train_tensor.reshape(-1, 1, X1_train_tensor.shape[1], 1)
    X2_test_tensor1 = X2_test_tensor.reshape(-1, 1, X2_test_tensor.shape[1], 1)
    X2_train_tensor1 = X2_train_tensor.reshape(-1, 1, X2_train_tensor.shape[1], 1)
    # y_train_tensor1 = y_train_tensor.reshape(-1,1,1,1)
    # y_test_tensor1 = y_test_tensor.reshape(-1,1,1,1)
    # 创建TensorDataset和DataLoader

    train1_dataset = TensorDataset(X1_train_tensor1, y1_train_tensor)
    test1_dataset = TensorDataset(X1_test_tensor1, y1_test_tensor)
    train2_dataset = TensorDataset(X2_train_tensor1, y2_train_tensor)
    test2_dataset = TensorDataset(X2_test_tensor1, y2_test_tensor)

    if partition_id == 1:
        train_loader = DataLoader(dataset=train1_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=test1_dataset, batch_size=256, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train2_dataset, batch_size=256, shuffle=True)
        test_loader = DataLoader(dataset=test2_dataset, batch_size=256, shuffle=False)
    return train_loader, test_loader
