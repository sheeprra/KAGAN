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

from tqdm import tqdm  
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

import torch_geometric

from sklearn.model_selection import StratifiedKFold

class GNNConv(MessagePassing):
    def __init__(self, in_feats, out_feats, alpha, drop_prob=0.0):
        super().__init__(aggr=None)  
        self.drop_prob = drop_prob
        self.kan = KAN([in_feats,64, out_feats])  
        self.a = nn.Parameter(torch.zeros(size=(2*out_feats, 1)))
        self.leakrelu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.a)

    def forward(self, x, edge_index):
      
        h = self.kan(x)

        return h

    def message(self, x_i, x_j, edge_index_i):

        return x_i
    




def get_endNum(str1):
    return int(str1.split('/')[-1])

def change_t1t2(file):
    return file.replace('T1','T2')

def change_file(file):
    return file.replace('1','2')


np.random.seed(0)


match_list_t1 = []
match_list_t2 = []

num_list = []
for root,dir,files in os.walk("/data/brain_all_data_TeZhengTiQu/AD/T1"):
    for file in files:
        
        num_list.append(get_endNum(root))

        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        
        match_list_t1.append(df.iloc[:,1].values.tolist())

       
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
      
        match_list_t2.append(df.iloc[:,1].values.tolist())

for root,dir,files in os.walk("/data/brain_all_data_TeZhengTiQu/CN/T1"):
    for file in files:
        num_list.append(get_endNum(root))
        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        match_list_t1.append(df.iloc[:,1].values.tolist())

    
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
        
        match_list_t2.append(df.iloc[:,1].values.tolist())

for root,dir,files in os.walk("/data/brain_all_data_TeZhengTiQu/MCI/T1"):
    for file in files:
        num_list.append(get_endNum(root))
        
        df = pd.read_csv(os.path.join(root,file), header=None, encoding='gbk').reset_index(drop=True)
        
        match_list_t1.append(df.iloc[:,1].values.tolist())

 
        df = pd.read_csv(os.path.join(change_t1t2(root),change_file(file)), header=None, encoding='gbk').reset_index(drop=True)
      
        match_list_t2.append(df.iloc[:,1].values.tolist())

data_array_t1 = np.array(match_list_t1)
data_array_t2 = np.array(match_list_t2)
num_array = np.array(num_list)


indices = np.arange(len(data_array_t1))

np.random.shuffle(indices)

data_array_t1 = data_array_t1[indices]
data_array_t2 = data_array_t2[indices]
num_array = num_array[indices]
print("flag2category",num_array)


t1_corr_matrix = np.corrcoef(data_array_t1)


t2_corr_matrix = np.corrcoef(data_array_t2)

t1_t2_matrix = t1_corr_matrix * t2_corr_matrix

labels = np.array([ 0 if x < 32 else 1 if x < 64 else 2 for x in num_array])



data = pd.read_csv('/data/brain_all_data_Init/dataT1.csv', header=None)
data = pd.read_csv('/data/brain_all_data_Init/dataT2.csv', header=None)

print("data shape",data.shape)


k = 5  
edge_index = [[], []]  
t1_t2_matrix = np.array(t1_t2_matrix)
for i in range(t1_t2_matrix.shape[0]):
   
    neighbors = np.argsort(t1_t2_matrix[i, :])[-(k+1):-1]  
    for neighbor in neighbors:
        edge_index[0].append(i)      
        edge_index[1].append(neighbor) 

edge_index = torch.tensor(edge_index, dtype=torch.long)
print("create edge",edge_index.shape)


x = torch.tensor(data, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)

data = Data(x=x, edge_index=edge_index, y=y)

print("Graph data",data)

labels = data.y.cpu().numpy()


best_Inf = []


class GraphNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels,alpha=0.2):
        super(GraphNet, self).__init__()
        self.conv1 = GNNConv(in_feats=in_channels, out_feats=116,alpha=0.2,drop_prob=0.2)
        self.conv2 = GNNConv(in_feats=116,out_feats = out_channels,alpha=0.2,drop_prob=0.0)

    def forward(self, x, edge_index):
        x, edge_index = data.x, data.edge_index

        x = F.normalize(x, p=2.0, dim=-1)

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)



device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(kf.split(data.x, labels)):
    print(f"\nFold {fold + 1}/{kf.n_splits}")
    print("*" * 90)


    model = GraphNet(in_channels=data.num_features, out_channels=3, alpha=0.2)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001 , weight_decay=5e-4)
 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                               T_max=50)

    best_acc = 0.0
    best_metrics = {}
    best_epoch = 0


    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    

    train_data = Data(x=data.x, edge_index=data.edge_index, y=data.y)
    train_data.train_mask = train_mask
    train_data.test_mask = test_mask


    train_loader = torch_geometric.data.DataLoader([train_data], batch_size=1, shuffle=True)
    test_loader = torch_geometric.data.DataLoader([train_data], batch_size=1, shuffle=False)

    model = model.to(device)


    for epoch in tqdm(range(200), desc="Training Progress"):
        model.train()
        optimizer.zero_grad()
        acc_train = 0.0

        for batch in train_loader:
            out = model(batch.x, batch.edge_index)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()
            pred = out[batch.test_mask].argmax(dim=1)
            y_true = batch.y[batch.test_mask].cpu().numpy()
            y_pred = pred.cpu().numpy()
            acc_train = accuracy_score(y_true, y_pred)

        scheduler.step()

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                out = model(batch.x, batch.edge_index)
                pred = out[batch.test_mask].argmax(dim=1)
                y_true = batch.y[batch.test_mask].cpu().numpy()
                y_pred = pred.cpu().numpy()

                acc = accuracy_score(y_true, y_pred)
                sen = recall_score(y_true, y_pred, average='macro')  
                pre = precision_score(y_true, y_pred, average='macro') 
                f1 = f1_score(y_true, y_pred, average='macro')  
                try:
                    auc = roc_auc_score(y_true, F.softmax(out[batch.test_mask], dim=1).cpu().numpy(), multi_class='ovo')  
                except ValueError:
                    auc = float('nan')  

    
                print(f"Epoch {epoch + 1}, acc_train:{acc_train:.4f}, Loss: {loss.item():.4f}, Accuracy: {acc:.4f}, Sen: {sen:.4f}, Pre: {pre:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")


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
