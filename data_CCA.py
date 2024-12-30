import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import os

match_list_t1 = []
match_list_t2 = []

num_list = []
for i in num_list:
    if i<=943:
        root_t1 = "/data/brain_all_data_TeZhengTiQu/AD/T1/"+str(i)
        root_t2 = "/data/brain_all_data_TeZhengTiQu/AD/T2/"+str(i)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)


        t1_features = pd_t1.iloc[:, [1, 6, 7]].values 
        t2_features = pd_t2.iloc[:, [1, 2]].values   
       
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))
     
        cca.fit(t1_features, t2_features)

        t1_cca, t2_cca = cca.transform(t1_features, t2_features)

        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
    
        combined_features = np.hstack((t1_cca, t2_cca))

    
        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')

    elif i<=1827:
        root_t1 = "/data/brain_all_data_TeZhengTiQu/CN/T1/"+str(i)
        root_t2 = "/data/brain_all_data_TeZhengTiQu/CN/T2/"+str(i)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
      

        t1_features = pd_t1.iloc[:, [1, 6, 7]].values 
        t2_features = pd_t2.iloc[:, [1, 2]].values    
      
 
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))

        cca.fit(t1_features, t2_features)


        t1_cca, t2_cca = cca.transform(t1_features, t2_features)
       
        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
      
        combined_features = np.hstack((t1_cca, t2_cca))

      
        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')

    else:
        root_t1 = "/data/brain_all_data_TeZhengTiQu/MCI/T1/"+str(i)
        root_t2 = "/data/brain_all_data_TeZhengTiQu/MCI/T2/"+str(i)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        
            
        
        t1_features = pd_t1.iloc[:, [1, 6, 7]].values  
        t2_features = pd_t2.iloc[:, [1, 2]].values    
       
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))

        cca.fit(t1_features, t2_features)

      
        t1_cca, t2_cca = cca.transform(t1_features, t2_features)
       
        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
      
        combined_features = np.hstack((t1_cca, t2_cca))


        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')




features_T1 = np.random.rand(116, 3)  
features_T2 = np.random.rand(116, 2)  

cca = CCA(n_components=min(features_T1.shape[1], features_T2.shape[1]))  
T1_canonical, T2_canonical = cca.fit_transform(features_T1, features_T2)
print(T1_canonical.shape, T2_canonical.shape)

alpha = 0.5 
T1_weighted = alpha * T1_canonical
T2_weighted = (1 - alpha) * T2_canonical


features_combined = T1_weighted + T2_weighted

print("Combined Features Shape:", features_combined.shape)
