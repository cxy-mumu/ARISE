import argparse
import os
import warnings
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_curve, auc

from dataset import load_data5, load_data2choose, load_data2



# #############################################################################
# 1. Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ff = 0.0
conf_matrix=0.0
recall=0.0
precision=0.0
f1=0.0
fpr = dict()
tpr = dict()
roc_auc = dict()
all_labels_true=0.0
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, padding=1):
        super(ResidualBlock, self).__init__()
        # 第一个卷积层使用输入通道数和步长
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层使用输出通道数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 如果输入输出通道不一致，使用1x1卷积来匹配维度
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.shortcut is not None:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)
        return out
class CNNModel(nn.Module):
    def __init__(self, num_classes, num_features):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # 调整池化层参数
        self.res_block1 = ResidualBlock(32, 64, stride=2, padding=1)
        self.pool = nn.MaxPool2d(2, 2, ceil_mode=True)  # 使用ceil_mode确保输出尺寸不为0
        self.dropout1 = nn.Dropout(0.3)
        # 调整全连接层的输入特征数
        self.fc1 = nn.Linear(1920, 1024)   # data5

        # self.fc1 = nn.Linear(896, 1024)  # data2
        # self.fc1 = nn.Linear(640, 1024)  # data2choose

        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)


    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))


        x = self.res_block1(x)
        x = self.pool(x)
        x = self.dropout1(x)



        N, C, H, W = x.size()  # 获取x的尺寸
        flattened_size = C * H * W  # 计算展平后的尺寸

        x = x.view(-1, flattened_size)  # 展平层
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout1(x)
        x = self.fc3(x)
        # x = self.fc2(x)
        return x



def train(model, train_loader, epochs,optimizer,criterion):
    """Train the model on the training set."""
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式
        running_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(images)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播
            loss.backward()
            # 优化
            optimizer.step()

            # 计算损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

        # 计算平均损失和准确率
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

def test(model, test_loader, criterion):
    """Validate the model on the test set."""
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss = 0.0
        all_outputs = []
        predictions = []
        true_labels = []
        for images, labels in test_loader:
            outputs = model(images)
            losses = criterion(outputs, labels)
            loss += losses.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            probabilities = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
            if isinstance(probabilities, torch.Tensor):
                probabilities = probabilities.cpu().numpy()
            all_outputs.extend(probabilities)
    conf_matrix = confusion_matrix(true_labels, predictions)
    recall = recall_score(true_labels, predictions, average='weighted')  # 可以指定 'micro' 或 'macro' 或 'weighted'
    precision = precision_score(true_labels, predictions, average='weighted')
    f1 = f1_score(true_labels, predictions, average='weighted')
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    all_outputs = np.array(all_outputs)
    true_labels = np.array(true_labels)
    n_classes = all_outputs.shape[1]
    for i in range(n_classes):
        y_true = (true_labels == i)
        if not np.any(y_true):  # 检查是否有任何正样本
            print(f"Warning: No positive samples for class {i}")
            continue
        fpr[i], tpr[i], _ = roc_curve(y_true, all_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    avg_loss = loss / len(test_loader)
    acc = correct / total
    print("avg_loss:", avg_loss, "acc:", acc)
    return avg_loss, acc, conf_matrix, recall, precision, f1, fpr,tpr,roc_auc


num_features = 12  # 获取编码后的特征数量
num_classes = 5  # 假设您有5个类别

model = CNNModel(num_classes, num_features)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################

# Get partition id
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    # choices=[1, 2],     #私有数据集
    choices=[1, 2, 3, 4, 5],   #公共数据集
    required=True,
    type=int,
    help="Partition of the dataset divided into 3 iid partitions created artificially.",
)
partition_id = parser.parse_args().partition_id

# Load model and data
net = CNNModel(num_classes, num_features).to(DEVICE)
#公共
trainloader, testloader = load_data5(partition_id=partition_id)
#私有
# trainloader, testloader = load_data2(partition_id=partition_id)
# trainloader, testloader = load_data2choose(partition_id=partition_id)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    best_accuracy = 0
    best_loss = float('inf')
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        for _ in range(1):
            train(net, trainloader, epochs=5, optimizer=optimizer, criterion=nn.CrossEntropyLoss())
            # model_save_path = os.path.join('model', 'quan', f'client_{partition_id}_round_{config["round"]}.pth')
            model_save_path = os.path.join('model', 'c1', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'c1c2', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'c1c2res', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'c1c2resf1', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'c1c2resf1f2', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'c2resf1f2f3', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'resf1f2f3', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'f1f2f3', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'f2f3', f'client_{partition_id}_round_{config["round"]}.pth')
            # model_save_path = os.path.join('model', 'f3', f'client_{partition_id}_round_{config["round"]}.pth')
            torch.save(net.state_dict(), model_save_path)
            print(f'Model saved to {model_save_path}')

        return self.get_parameters(config={}), len(trainloader.dataset), {}


    def get_parameters(self, config):
        # 假设我们只上传 conv1 和 fc1 层的参数
        params_to_upload = {
            'conv1.weight': net.conv1.weight.detach().cpu().numpy(),
            'conv1.bias': net.conv1.bias.detach().cpu().numpy(),
            'bn1.weight': net.bn1.weight.detach().cpu().numpy(),
            'bn1.bias': net.bn1.bias.detach().cpu().numpy(),
            'bn1.running_mean': net.bn1.running_mean.detach().cpu().numpy(),
            'bn1.running_var': net.bn1.running_var.detach().cpu().numpy(),
            #
            # 'conv2.weight': net.conv2.weight.detach().cpu().numpy(),
            # 'conv2.bias': net.conv2.bias.detach().cpu().numpy(),
            # 'bn2.weight': net.bn2.weight.detach().cpu().numpy(),
            # 'bn2.bias': net.bn2.bias.detach().cpu().numpy(),
            # 'bn2.running_mean': net.bn2.running_mean.detach().cpu().numpy(),
            # 'bn2.running_var': net.bn2.running_var.detach().cpu().numpy(),
            # # # #
            # 'res_block1.conv1.weight': net.res_block1.conv1.weight.detach().cpu().numpy(),
            # # 'res_block1.conv1.bias': net.res_block1.conv1.bias.detach().cpu().numpy()if hasattr(net.res_block1,
            # #                                                                                      'conv1') else None,
            # 'res_block1.bn1.weight': net.res_block1.bn1.weight.detach().cpu().numpy() if hasattr(net.res_block1,
            #                                                                                      'bn1') else None,
            # 'res_block1.bn1.bias': net.res_block1.bn1.bias.detach().cpu().numpy() if hasattr(net.res_block1,
            #                                                                                  'bn1') else None,
            # 'res_block1.bn1.running_mean': net.res_block1.bn1.running_mean.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'bn1') else None,
            # 'res_block1.bn1.running_var': net.res_block1.bn1.running_var.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'bn1') else None,
            #
            # 'res_block1.conv2.weight': net.res_block1.conv2.weight.detach().cpu().numpy(),
            # # 'res_block1.conv2.bias': net.res_block1.conv2.bias.detach().cpu().numpy() if hasattr(net.res_block1,
            # #                                                                                      'conv2') else None,
            # 'res_block1.bn2.weight': net.res_block1.bn2.weight.detach().cpu().numpy() if hasattr(net.res_block1,
            #                                                                                      'bn2') else None,
            # 'res_block1.bn2.bias': net.res_block1.bn2.bias.detach().cpu().numpy() if hasattr(net.res_block1,
            #                                                                                  'bn2') else None,
            # 'res_block1.bn2.running_mean': net.res_block1.bn2.running_mean.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'bn2') else None,
            # 'res_block1.bn2.running_var': net.res_block1.bn2.running_var.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'bn2') else None,
            # #
            # 'res_block1.shortcut.0.weight': net.res_block1.shortcut[0].weight.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'shortcut') and len(net.res_block1.shortcut) > 0 else None,
            # 'res_block1.shortcut.1.weight': net.res_block1.shortcut[1].weight.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'shortcut') and len(net.res_block1.shortcut) > 1 else None,
            # 'res_block1.shortcut.1.bias': net.res_block1.shortcut[1].bias.detach().cpu().numpy() if hasattr(
            #     net.res_block1, 'shortcut') and len(net.res_block1.shortcut) > 1 else None,
            # 'res_block1.shortcut.1.running_mean': net.res_block1.shortcut[
            #     1].running_mean.detach().cpu().numpy() if hasattr(net.res_block1, 'shortcut') and len(
            #     net.res_block1.shortcut) > 1 else None,
            # 'res_block1.shortcut.1.running_var': net.res_block1.shortcut[
            #     1].running_var.detach().cpu().numpy() if hasattr(net.res_block1, 'shortcut') and len(
            #     net.res_block1.shortcut) > 1 else None,
            # # # #
            # 'fc1.weight': net.fc1.weight.detach().cpu().numpy(),
            # 'fc1.bias': net.fc1.bias.detach().cpu().numpy(),
            # 'fc2.weight': net.fc2.weight.detach().cpu().numpy(),
            # 'fc2.bias': net.fc2.bias.detach().cpu().numpy(),
            # 'fc3.weight': net.fc3.weight.detach().cpu().numpy(),
            # 'fc3.bias': net.fc3.bias.detach().cpu().numpy(),
        }

        return list(params_to_upload.values())


    def set_parameters(self, parameters):
        # 将参数从列表转换回字典
        # 将 NumPy 数组转换为元组
        ####c1
        parmname =['conv1.weight','conv1.bias','bn1.weight','bn1.bias'
                   'bn1.running_mean','bn1.running_var']

        # c1c2
        # parmname =['conv1.weight','conv1.bias','bn1.weight','bn1.bias',
        #            'bn1.running_mean','bn1.running_var',
        #            'conv2.weight','conv2.bias','bn2.weight','bn2.bias'
        #            'bn2.running_mean','bn2.running_var']
        # c1c2res
        # parmname = ['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias',
        #             'bn1.running_mean', 'bn1.running_var',
        #             'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
        #             'bn2.running_mean', 'bn2.running_var',
        #             'res_block1.conv1.weight',
        #             'res_block1.bn1.weight', 'res_block1.bn1.bias',
        #             'res_block1.bn1.running_mean', 'res_block1.bn1.running_var',
        #             'res_block1.conv2.weight',
        #             'res_block1.bn2.weight', 'res_block1.bn2.bias',
        #             'res_block1.bn2.running_mean', 'res_block1.bn2.running_var',
        #             'res_block1.shortcut.0.weight', 'res_block1.shortcut.1.weight',
        #             'res_block1.shortcut.1.bias', 'res_block1.shortcut.1.running_mean',
        #             'res_block1.shortcut.1.running_var']
            # 假设shortcut有两个卷积层，第一个只有权重，第二个有权重、偏置、均值和方差
        # c1c2resf1
        # parmname =['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias',
        #            'bn1.running_mean', 'bn1.running_var',
        #            'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
        #            'bn2.running_mean', 'bn2.running_var',
        #            'res_block1.conv1.weight',
        #            'res_block1.bn1.weight', 'res_block1.bn1.bias',
        #            'res_block1.bn1.running_mean', 'res_block1.bn1.running_var',
        #            'res_block1.conv2.weight',
        #            'res_block1.bn2.weight', 'res_block1.bn2.bias',
        #            'res_block1.bn2.running_mean', 'res_block1.bn2.running_var',
        #            'res_block1.shortcut.0.weight', 'res_block1.shortcut.1.weight',
        #            'res_block1.shortcut.1.bias', 'res_block1.shortcut.1.running_mean',
        #            'res_block1.shortcut.1.running_var'
        #            'fc1.weight','fc1.bias']
        # c1c2resf1f2
        # parmname =['conv1.weight', 'conv1.bias', 'bn1.weight', 'bn1.bias',
        #            'bn1.running_mean', 'bn1.running_var',
        #            'conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
        #            'bn2.running_mean', 'bn2.running_var',
        #            'res_block1.conv1.weight',
        #            'res_block1.bn1.weight', 'res_block1.bn1.bias',
        #            'res_block1.bn1.running_mean', 'res_block1.bn1.running_var',
        #            'res_block1.conv2.weight',
        #            'res_block1.bn2.weight', 'res_block1.bn2.bias',
        #            'res_block1.bn2.running_mean', 'res_block1.bn2.running_var',
        #            'res_block1.shortcut.0.weight', 'res_block1.shortcut.1.weight',
        #            'res_block1.shortcut.1.bias', 'res_block1.shortcut.1.running_mean',
        #            'res_block1.shortcut.1.running_var'
        #            'fc1.weight','fc1.bias'
        #            'fc2.weight','fc2.bias']


        # c2resf1f2f3
        # parmname =['conv2.weight', 'conv2.bias', 'bn2.weight', 'bn2.bias',
        #            'bn2.running_mean', 'bn2.running_var',
        #            'res_block1.conv1.weight',
        #            'res_block1.bn1.weight', 'res_block1.bn1.bias',
        #            'res_block1.bn1.running_mean', 'res_block1.bn1.running_var',
        #            'res_block1.conv2.weight',
        #            'res_block1.bn2.weight', 'res_block1.bn2.bias',
        #            'res_block1.bn2.running_mean', 'res_block1.bn2.running_var',
        #            'res_block1.shortcut.0.weight', 'res_block1.shortcut.1.weight',
        #            'res_block1.shortcut.1.bias', 'res_block1.shortcut.1.running_mean',
        #            'res_block1.shortcut.1.running_var'
        #            'fc1.weight','fc1.bias'
        #            'fc2.weight','fc2.bias'
        #            'fc3.weight','fc3.bias']

        # # resf1f2f3
        # parmname =['res_block1.conv1.weight',
        #            'res_block1.bn1.weight', 'res_block1.bn1.bias',
        #            'res_block1.bn1.running_mean', 'res_block1.bn1.running_var',
        #            'res_block1.conv2.weight',
        #            'res_block1.bn2.weight', 'res_block1.bn2.bias',
        #            'res_block1.bn2.running_mean', 'res_block1.bn2.running_var',
        #            'res_block1.shortcut.0.weight', 'res_block1.shortcut.1.weight',
        #            'res_block1.shortcut.1.bias', 'res_block1.shortcut.1.running_mean',
        #            'res_block1.shortcut.1.running_var'
        #            'fc1.weight','fc1.bias'
        #            'fc2.weight','fc2.bias'
        #            'fc3.weight','fc3.bias']
        # f1f2f3
        # parmname =['fc1.weight','fc1.bias','fc2.weight','fc2.bias','fc3.weight','fc3.bias']
        # f2f3
        # parmname =['fc2.weight','fc2.bias','fc3.weight','fc3.bias']
        # f3
        # parmname =['fc3.weight','fc3.bias']

        # parmname = []

        # print("123:",parmname,"par:",parameters)
        params_dict = dict(zip(parmname, parameters))

        # 更新模型参数，使用 requires_grad=False
        if 'conv1.weight' in params_dict:
            net.conv1.weight.data = torch.tensor(params_dict['conv1.weight'], dtype=torch.float32).requires_grad_(False)
            net.conv1.bias.data = torch.tensor(params_dict['conv1.bias'], dtype=torch.float32).requires_grad_(False)
            if 'bn1' in params_dict:
                net.bn1.weight.data = torch.tensor(params_dict['bn1.weight'], dtype=torch.float32).requires_grad_(False)
                net.bn1.bias.data = torch.tensor(params_dict['bn1.bias'], dtype=torch.float32).requires_grad_(False)
                net.bn1.running_mean = torch.tensor(params_dict['bn1.running_mean'], dtype=torch.float32)
                net.bn1.running_var = torch.tensor(params_dict['bn1.running_var'], dtype=torch.float32)

        # 确保conv2和bn2层存在
        if 'conv2.weight' in params_dict:
            net.conv2.weight.data = torch.tensor(params_dict['conv2.weight'], dtype=torch.float32).requires_grad_(False)
            net.conv2.bias.data = torch.tensor(params_dict['conv2.bias'], dtype=torch.float32).requires_grad_(False)
            if 'bn2' in params_dict:
                net.bn2.weight.data = torch.tensor(params_dict['bn2.weight'], dtype=torch.float32).requires_grad_(False)
                net.bn2.bias.data = torch.tensor(params_dict['bn2.bias'], dtype=torch.float32).requires_grad_(False)
                net.bn2.running_mean = torch.tensor(params_dict['bn2.running_mean'], dtype=torch.float32)
                net.bn2.running_var = torch.tensor(params_dict['bn2.running_var'], dtype=torch.float32)

        # 确保res_block1层存在
        if 'res_block1' in params_dict:
            # 确保res_block1.conv1层存在
            if 'res_block1.conv1.weight' in params_dict:
                net.res_block1.conv1.weight.data = torch.tensor(params_dict['res_block1.conv1.weight'],
                                                                dtype=torch.float32).requires_grad_(False)
                net.res_block1.conv1.bias.data = torch.tensor(params_dict['res_block1.conv1.bias'],
                                                              dtype=torch.float32).requires_grad_(False)

            # 确保res_block1.bn1层存在
            if 'res_block1.bn1' in params_dict and params_dict['res_block1.bn1'] is not None:
                bn1_params = params_dict['res_block1.bn1']
                net.res_block1.bn1.weight.data = torch.tensor(bn1_params.get('weight', None),
                                                              dtype=torch.float32).requires_grad_(False)
                net.res_block1.bn1.bias.data = torch.tensor(bn1_params.get('bias', None),
                                                            dtype=torch.float32).requires_grad_(False)
                net.res_block1.bn1.running_mean = torch.tensor(bn1_params.get('running_mean', None),
                                                               dtype=torch.float32)
                net.res_block1.bn1.running_var = torch.tensor(bn1_params.get('running_var', None), dtype=torch.float32)

            # 确保res_block1.conv2层存在
            if 'res_block1.conv2.weight' in params_dict:
                net.res_block1.conv2.weight.data = torch.tensor(params_dict['res_block1.conv2.weight'],
                                                                dtype=torch.float32).requires_grad_(False)
                net.res_block1.conv2.bias.data = torch.tensor(params_dict['res_block1.conv2.bias'],
                                                              dtype=torch.float32).requires_grad_(False)

            # 确保res_block1.bn2层存在
            if 'res_block1.bn2' in params_dict and params_dict['res_block1.bn2'] is not None:
                bn2_params = params_dict['res_block1.bn2']
                net.res_block1.bn2.weight.data = torch.tensor(bn2_params.get('weight', None),
                                                              dtype=torch.float32).requires_grad_(False)
                net.res_block1.bn2.bias.data = torch.tensor(bn2_params.get('bias', None),
                                                            dtype=torch.float32).requires_grad_(False)
                net.res_block1.bn2.running_mean = torch.tensor(bn2_params.get('running_mean', None),
                                                               dtype=torch.float32)
                net.res_block1.bn2.running_var = torch.tensor(bn2_params.get('running_var', None), dtype=torch.float32)

            # 确保res_block1.shortcut层存在
            if 'res_block1.shortcut' in params_dict and params_dict['res_block1.shortcut'] is not None:
                shortcut_params = params_dict['res_block1.shortcut']
                if '0.weight' in shortcut_params:
                    net.res_block1.shortcut[0].weight.data = torch.tensor(shortcut_params['0.weight'],
                                                                          dtype=torch.float32).requires_grad_(False)
                if '1.weight' in shortcut_params and '1.bias' in shortcut_params:
                    net.res_block1.shortcut[1].weight.data = torch.tensor(shortcut_params['1.weight'],
                                                                          dtype=torch.float32).requires_grad_(False)
                    net.res_block1.shortcut[1].bias.data = torch.tensor(shortcut_params['1.bias'],
                                                                        dtype=torch.float32).requires_grad_(False)
                    net.res_block1.shortcut[1].running_mean = torch.tensor(shortcut_params.get('1.running_mean', None),
                                                                           dtype=torch.float32)
                    net.res_block1.shortcut[1].running_var = torch.tensor(shortcut_params.get('1.running_var', None),
                                                                          dtype=torch.float32)

        # 确保fc层存在
        if 'fc1.weight' in params_dict:
            net.fc1.weight.data = torch.tensor(params_dict['fc1.weight'], dtype=torch.float32).requires_grad_(False)
            net.fc1.bias.data = torch.tensor(params_dict['fc1.bias'], dtype=torch.float32).requires_grad_(False)
            # 以此类推，更新fc2和fc3层的参数
        if 'fc2.weight' in params_dict:
            net.fc2.weight.data = torch.tensor(params_dict['fc2.weight'], dtype=torch.float32).requires_grad_(False)
            net.fc2.bias.data = torch.tensor(params_dict['fc2.bias'], dtype=torch.float32).requires_grad_(False)
        if 'fc3.weight' in params_dict:
            net.fc3.weight.data = torch.tensor(params_dict['fc3.weight'], dtype=torch.float32).requires_grad_(False)
            net.fc3.bias.data = torch.tensor(params_dict['fc3.bias'], dtype=torch.float32).requires_grad_(False)


    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, c, r, p, f, fprr, tprr, roc_aucc= test(net, testloader,criterion = nn.CrossEntropyLoss())
        if accuracy >= self.best_accuracy:
            self.best_accuracy = accuracy
            global ff
            global conf_matrix
            global recall
            global precision
            global f1
            global fpr
            global tpr
            global roc_auc
            global all_labels_true
            ff = accuracy
            conf_matrix=c
            recall=r
            precision=p
            f1=f
            fpr=fprr
            tpr=tprr
            roc_auc=roc_aucc
            # 构建文件路径
        return loss, len(testloader.dataset), {"accuracy": accuracy}





# Start Flower client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient().to_client(),
)

print("Max_Accuracy：", ff)
print("recall：", recall)
print("precision", precision)
print("f1：", f1)
