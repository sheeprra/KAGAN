import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax, add_remaining_self_loops
import numpy as np
import pandas as pd

import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from efficient_kan import KAN
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

import os
import pandas as pd


import torch
import numpy as np
from torch_geometric.data import Data

from torch_geometric.nn import GCNConv
import torch.nn.functional as F

from tqdm import tqdm  # 引入 tqdm 进度条
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import torch_geometric
# 使用StratifiedKFold进行划分
from sklearn.model_selection import StratifiedKFold

class GNNConv(MessagePassing):
    def __init__(self, in_feats, out_feats, alpha, drop_prob=0.0):
        super().__init__(aggr=None)  # 取消聚合
        self.drop_prob = drop_prob
        self.kan = KAN([in_feats,64, out_feats])  # 替换线性变换为 KAN
        self.a = nn.Parameter(torch.zeros(size=(2*out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
        # 取消自环添加
        # edge_index, _ = add_remaining_self_loops(edge_index)
        
        # 计算 KAN 变换
        h = self.kan(x)
        
        # 直接返回 KAN 输出，无邻居聚合
        return h

    # 由于没有邻居聚合，因此不再需要 message 函数
    # 但为了保持框架完整性，可以保留 message 函数的定义
    def message(self, x_i, x_j, edge_index_i):
        # 这里可以直接返回 x_i，无需考虑邻居信息
        return x_i
    




def get_endNum(str1):
    return int(str1.split('/')[-1])

def change_t1t2(file):
    return file.replace('T1','T2')

def change_file(file):
    return file.replace('1','2')

# 设置随机种子
np.random.seed(0)

#存储特征值
match_list_t1 = []
match_list_t2 = []
#存储特征值对应的数字编号
num_list = []
for root,dir,files in os.walk("/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/AD/T1"):
    for file in files:
        
        num_list.append(get_endNum(root))
        # print(get_endNum(root))
        # print(os.path.join(root,file))
        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t1.append(df.iloc[:,1].values.tolist())

        #同样方式处理T2
        #将root .split('/')[-2] T1->T2
        #将路径file中的1换成2
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t2.append(df.iloc[:,1].values.tolist())

for root,dir,files in os.walk("/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/CN/T1"):
    for file in files:
        num_list.append(get_endNum(root)+32)
        # print(get_endNum(root))
        # print(os.path.join(root,file))
        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t1.append(df.iloc[:,1].values.tolist())

        #同样方式处理T2      
        #将路径file中的1换成2
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t2.append(df.iloc[:,1].values.tolist())

for root,dir,files in os.walk("/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/MCI/T1"):
    for file in files:
        num_list.append(get_endNum(root)+64)
        # print(get_endNum(root))
        # print(os.path.join(root,file))
        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t1.append(df.iloc[:,1].values.tolist())

        #同样方式处理T2
        #将路径file中的1换成2
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
        #获取df第二列信息,并将其转化为list
        match_list_t2.append(df.iloc[:,1].values.tolist())

#根据match_list计算persion相关系数 
# 转换为 numpy 数组
data_array_t1 = np.array(match_list_t1)
data_array_t2 = np.array(match_list_t2)
num_array = np.array(num_list)

# 创建索引数组
indices = np.arange(len(data_array_t1))
# 打乱索引数组
np.random.shuffle(indices)
# 使用打乱的索引同时打乱两个数组
data_array_t1 = data_array_t1[indices]
data_array_t2 = data_array_t2[indices]
num_array = num_array[indices]
print("flag2category",num_array)

# np.random.shuffle(data_array)
# 计算 T1 信息的 Pearson 相关系数矩阵
t1_corr_matrix = np.corrcoef(data_array_t1)

# 计算 Pearson 相关系数矩阵
t2_corr_matrix = np.corrcoef(data_array_t2)

#求t1_corr_matrix与t2_corr_matrix的哈达玛积
#哈达玛积 求取邻接矩阵
t1_t2_matrix = t1_corr_matrix * t2_corr_matrix

# 测试： 0-31: AD, 32-63: CN, 64-95: MCI
#根据对应的数字编号，将其转化为标签
labels = np.array([ 0 if x < 32 else 1 if x < 64 else 2 for x in num_array])

# 加载数据
# data = pd.read_csv('/home/tangwenhao/TRC/data/brain_all_data_CCA/data.csv', header=None)
# data = data.values.reshape(96, 4 * 116)  # 将11136行重新 reshape 成 (96, 4*116)

data = pd.read_csv('/home/tangwenhao/TRC/data/brain_all_data_Init/dataT1.csv', header=None)
data = data.values.reshape(96, 3 * 116)  # 将11136行重新 reshape 成 (96, 3*116)

data = pd.read_csv('/home/tangwenhao/TRC/data/brain_all_data_Init/dataT2.csv', header=None)
data = data.values.reshape(96, 2 * 116)  # 将11136行重新 reshape 成 (96, 3*116)
print("data shape",data.shape)

# 2. 获取每个节点最大的5个邻接点
k = 5  # 最大邻接点数
edge_index = [[], []]  # 用于存储边的索引（源节点和目标节点）
t1_t2_matrix = np.array(t1_t2_matrix)
for i in range(t1_t2_matrix.shape[0]):
    # 获取第i行的邻接值，排除自身！！！！！，并选择前5大的邻接点索引
    neighbors = np.argsort(t1_t2_matrix[i, :])[-(k+1):-1]  # 排序后选择最大的5个
    for neighbor in neighbors:
        edge_index[0].append(i)       # 源节点
        edge_index[1].append(neighbor)  # 目标节点
# 转换为 PyTorch Tensor 并作为 edge_index
edge_index = torch.tensor(edge_index, dtype=torch.long)
print("create edge",edge_index.shape)

# 3. 将特征数据和标签转换为 PyTorch Tensor
x = torch.tensor(data, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)
# print("x shape",x.shape)
# print("y shape",y.shape)

# 4. 使用 PyG 构建图数据结构
data = Data(x=x, edge_index=edge_index, y=y)
# 打印图数据
print("Graph data",data)


# 提取每个节点的标签
labels = data.y.cpu().numpy()


#记录全局训练信息
best_Inf = []


class GraphNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels,alpha=0.2):
        super(GraphNet, self).__init__()
        self.conv1 = GNNConv(in_feats=in_channels, out_feats=116,alpha=0.2,drop_prob=0.2)
        self.conv2 = GNNConv(in_feats=116,out_feats = out_channels,alpha=0.2,drop_prob=0.0)

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index
        #对x进行归一化
        x = F.normalize(x, p=2.0, dim=-1)
        # GCN第一层
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # GCN第二层
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


# 检查是否有可用的 GPU
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# 假设 data 是已经加载好的 PyG 数据对象，并将其移动到 GPU（如果有）
data = data.to(device)

from sklearn.model_selection import StratifiedKFold

# 创建分层五折交叉验证的实例
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 进行5折交叉验证
for fold, (train_idx, test_idx) in enumerate(kf.split(data.x, labels)):
    print(f"\nFold {fold + 1}/{kf.n_splits}")
    print("*" * 90)

    # 初始化模型
    model = GraphNet(in_channels=data.num_features, out_channels=3, alpha=0.2)
    
    # 使用随机梯度下降进行训练
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 , weight_decay=5e-4)
    # 定义学习率调度器
    # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=50)
    # 存储最佳模型信息
    best_acc = 0.0
    best_metrics = {}
    best_epoch = 0

    # 根据索引划分训练集和测试集
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    
    # 训练数据
    train_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
    train_data.train_mask = train_mask
    train_data.test_mask = test_mask

    # 创建 DataLoader 加载训练和验证数据
    train_loader = torch_geometric.data.DataLoader([train_data], batch_size=1, shuffle=True)
    test_loader = torch_geometric.data.DataLoader([train_data], batch_size=1, shuffle=False)

    # 将模型移动到 GPU
    model = model.to(device)

    # 训练和验证循环
    for epoch in tqdm(range(200), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        acc_train = 0.0
        # 训练
        for batch in train_loader:
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            pred = out[batch.test_mask].argmax(dim=1)
            y_true = batch.y[batch.test_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            acc_train = accuracy_score(y_true, y_pred)
            # 更新学习率
        scheduler.step()
        # 评估模型
        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index)
                pred = out[batch.test_mask].argmax(dim=1)
                y_true = batch.y[batch.test_mask].cpu().numpy()
                y_pred = pred.cpu().numpy()
                
                # 计算分类指标
                acc = accuracy_score(y_true, y_pred)
                sen = recall_score(y_true, y_pred, average='macro')  # 灵敏度（召回率）
                pre = precision_score(y_true, y_pred, average='macro')  # 精度
                f1 = f1_score(y_true, y_pred, average='macro')  # F1-score
                try:
                    auc = roc_auc_score(y_true, F.softmax(out[batch.test_mask], dim=1).cpu().numpy(), multi_class='ovo')  # AUC
                except ValueError:
                    auc = float('nan')  # 如果AUC无法计算，设置为NaN

                # 打印当前 epoch 的评估指标
                print(f"Epoch {epoch + 1}, acc_train:{acc_train:.4f}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Sen: {sen:.4f}, Pre: {pre:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

                # 如果准确率更高，保存当前的最佳模型信息
                if acc > best_acc:
                    best_metrics = {
                        'epoch': epoch + 1,
                        'accuracy': acc,
                        'sensitivity': sen,
                        'precision': pre,
                        'f1_score': f1,
                        'auc': auc,
                        'loss': loss.item(),
                    }
    best_Inf.append(best_metrics)
print(best_Inf)