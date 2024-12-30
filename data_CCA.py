import numpy as np
from sklearn.cross_decomposition import CCA
import pandas as pd
import os
# T1&T2 AD 0-31 CN 32 -63 MNI 64-95
# [27 92  4 49 84 94  6 72 41 68 63 69 71 12  7 26  9  5 54 25 32 58 24 73
#  44 38 23 85 70 64 33 67 95 79 45 15 29 37 53 74 22 46 43 42 20  3 30 57
#  55 19 81 47 18 36  1 61  0 56 17 60 11 21 66 34 80 50 93  8 75 13 14 62
#  78 90 83 28 91 89 76  2 51 40 39 86 59 16 87 52 31 77 10 65 88 82 35 48]


# 遍历数组[27 92  4 49 84 94  6 72 41 68 63 69 71 12  7 26  9  5 54 25 32 58 24 73
# 44 38 23 85 70 64 33 67 95 79 45 15 29 37 53 74 22 46 43 42 20  3 30 57
# 55 19 81 47 18 36  1 61  0 56 17 60 11 21 66 34 80 50 93  8 75 13 14 62
# 78 90 83 28 91 89 76  2 51 40 39 86 59 16 87 52 31 77 10 65 88 82 35 48]
# 如果数组为0-31，设置root_t1路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/AD/T1"+str(数组)，
# 设置root_t2路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/AD/T2"+str(数组),
# 如果数组为32-63，设置root_t1路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/CN/T1"+str(数组-32)，
# 设置root_t2路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/CN/T2"+str(数组-32),
# 如果数组为64-95，设置root_t1路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/MCI/T1"+str(数组-64)，
# 设置root_t2路径为"/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/MCI/T2"+str(数组-64),
match_list_t1 = []
match_list_t2 = []
#这里的num_list是一个数组，数组中的元素是0-95的数字，从Drow_DuiJiaoJuZhen.py中复制过来的
num_list = [27, 92,  4, 49, 84, 94,  6, 72, 41, 68, 63, 69, 71, 12,  7, 26,  9,  5, 54, 25, 32, 58, 24, 73,
    44, 38, 23, 85, 70, 64, 33, 67, 95, 79, 45, 15, 29, 37, 53, 74, 22, 46, 43, 42, 20,  3, 30, 57,
    55, 19, 81, 47, 18, 36,  1, 61,  0, 56, 17, 60, 11, 21, 66, 34, 80, 50, 93,  8, 75, 13, 14, 62,
    78, 90, 83, 28, 91, 89, 76,  2, 51, 40, 39, 86, 59, 16, 87, 52, 31, 77, 10, 65, 88, 82, 35, 48]
for i in num_list:
    if i<=31:
        root_t1 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/AD/T1/"+str(i)
        root_t2 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/AD/T2/"+str(i)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        # # 读取root_t1路径下的文件，将第二列、第七列、第八列信息转化为list，存储在match_list_t1中
        # match_list_t1.append(pd_t1.iloc[:,[1,6,7]].values.tolist())
        # # 读取root_t2路径下的文件，将第二列、第三列信息转化为list，存储在match_list_t2中，
        # match_list_t2.append(pd_t2.iloc[:,[1,2]].values.tolist())

        # 提取指定列进行典型相关性分析
        t1_features = pd_t1.iloc[:, [1, 6, 7]].values  # T1 特征: 第2, 第7, 第8列
        t2_features = pd_t2.iloc[:, [1, 2]].values    # T2 特征: 第2, 第3列
        # print(t1_features)
        # print(t2_features)
        # # 确保两个特征集的行数一致，典型相关性分析要求行数相等 这里是一致的，没必要写出来
        # min_length = min(len(t1_features), len(t2_features))
        # t1_features = t1_features[:min_length, :]
        # t2_features = t2_features[:min_length, :]

        # 创建 CCA 模型，n_components 是要保留的相关特征的维度数量 116
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))
        # 对两个特征集进行拟合
        cca.fit(t1_features, t2_features)

        # 进行典型相关性转换，得到两个新的特征集
        t1_cca, t2_cca = cca.transform(t1_features, t2_features)
        # print(t1_cca)
        # print(t2_cca)
        # 根据 alpha 参数调整特征权重比例
        # 假设我们对 t1_cca 和 t2_cca 的特征比例进行 alpha 加权
        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
        # # 输出结果
        # print("调整后的 T1 特征集（按 alpha 加权）：")
        # print(adjusted_t1_features)

        # print("调整后的 T2 特征集（按 1-alpha 加权）：")
        # print(adjusted_t2_features)
        
        # # 使用 numpy 的 vstack 函数进行垂直拼接 2*232
        # combined_features = np.vstack((t1_cca, t2_cca))
        # 使用 numpy 的 hstack 函数进行水平拼接
        combined_features = np.hstack((t1_cca, t2_cca))

        # print("拼接后的矩阵大小：", combined_features.shape)
        #将combined_features存储到brain_all_data_CCA文件夹下的data.csv文件中
        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/home/tangwenhao/TRC/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')

    elif i<=63:
        root_t1 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/CN/T1/"+str(i-32)
        root_t2 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/CN/T2/"+str(i-32)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        # # 读取root_t1路径下的文件，将第二列、第八列、第十列信息转化为list，存储在match_list_t1中
        # match_list_t1.append(pd_t1.iloc[:,[1,7,9]].values.tolist())
        # # 读取root_t2路径下的文件，将第二列、第三列信息转化为list，存储在match_list_t2中，
        # match_list_t2.append(pd_t2.iloc[:,[1,2]].values.tolist())
            
        # 提取指定列进行典型相关性分析
        t1_features = pd_t1.iloc[:, [1, 6, 7]].values  # T1 特征: 第2, 第7, 第8列
        t2_features = pd_t2.iloc[:, [1, 2]].values    # T2 特征: 第2, 第3列
        # print(t1_features)
        # print(t2_features)
        # # 确保两个特征集的行数一致，典型相关性分析要求行数相等 这里是一致的，没必要写出来
        # min_length = min(len(t1_features), len(t2_features))
        # t1_features = t1_features[:min_length, :]
        # t2_features = t2_features[:min_length, :]

        # 创建 CCA 模型，n_components 是要保留的相关特征的维度数量 116
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))
        # 对两个特征集进行拟合
        cca.fit(t1_features, t2_features)

        # 进行典型相关性转换，得到两个新的特征集
        t1_cca, t2_cca = cca.transform(t1_features, t2_features)
        # print(t1_cca)
        # print(t2_cca)
        # 根据 alpha 参数调整特征权重比例
        # 假设我们对 t1_cca 和 t2_cca 的特征比例进行 alpha 加权
        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
        # # 输出结果
        # print("调整后的 T1 特征集（按 alpha 加权）：")
        # print(adjusted_t1_features)

        # print("调整后的 T2 特征集（按 1-alpha 加权）：")
        # print(adjusted_t2_features)
        
        # # 使用 numpy 的 vstack 函数进行垂直拼接 2*232
        # combined_features = np.vstack((t1_cca, t2_cca))
        # 使用 numpy 的 hstack 函数进行水平拼接
        combined_features = np.hstack((t1_cca, t2_cca))

        # print("拼接后的矩阵大小：", combined_features.shape)
        #将combined_features存储到brain_all_data_CCA文件夹下的data.csv文件中
        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/home/tangwenhao/TRC/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')

    else:
        root_t1 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/MCI/T1/"+str(i-64)
        root_t2 = "/home/tangwenhao/TRC/data/brain_all_data_TeZhengTiQu/MCI/T2/"+str(i-64)

        root_tmp = os.listdir(root_t1)
        pd_t1 = pd.read_csv(os.path.join(root_t1,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        root_tmp = os.listdir(root_t2)
        pd_t2 = pd.read_csv(os.path.join(root_t2,root_tmp[0]), header=None, encoding='gbk').reset_index(drop=True)
        # # 读取root_t1路径下的文件，将第二列、第八列、第十列信息转化为list，存储在match_list_t1中
        # match_list_t1.append(pd_t1.iloc[:,[1,7,9]].values.tolist())
        # # 读取root_t2路径下的文件，将第二列、第三列信息转化为list，存储在match_list_t2中，
        # match_list_t2.append(pd_t2.iloc[:,[1,2]].values.tolist())
            
        # 提取指定列进行典型相关性分析
        t1_features = pd_t1.iloc[:, [1, 6, 7]].values  # T1 特征: 第2, 第7, 第8列
        t2_features = pd_t2.iloc[:, [1, 2]].values    # T2 特征: 第2, 第3列
        # print(t1_features)
        # print(t2_features)
        # # 确保两个特征集的行数一致，典型相关性分析要求行数相等 这里是一致的，没必要写出来
        # min_length = min(len(t1_features), len(t2_features))
        # t1_features = t1_features[:min_length, :]
        # t2_features = t2_features[:min_length, :]

        # 创建 CCA 模型，n_components 是要保留的相关特征的维度数量 116
        cca = CCA(n_components=min(t1_features.shape[1], t2_features.shape[1]))
        # 对两个特征集进行拟合
        cca.fit(t1_features, t2_features)

        # 进行典型相关性转换，得到两个新的特征集
        t1_cca, t2_cca = cca.transform(t1_features, t2_features)
        # print(t1_cca)
        # print(t2_cca)
        # 根据 alpha 参数调整特征权重比例
        # 假设我们对 t1_cca 和 t2_cca 的特征比例进行 alpha 加权
        alpha = 0.5
        adjusted_t1_features = alpha * t1_cca
        adjusted_t2_features = (1 - alpha) * t2_cca
        
        # # 输出结果
        # print("调整后的 T1 特征集（按 alpha 加权）：")
        # print(adjusted_t1_features)

        # print("调整后的 T2 特征集（按 1-alpha 加权）：")
        # print(adjusted_t2_features)
        
        # # 使用 numpy 的 vstack 函数进行垂直拼接 2*232
        # combined_features = np.vstack((t1_cca, t2_cca))
        # 使用 numpy 的 hstack 函数进行水平拼接
        combined_features = np.hstack((t1_cca, t2_cca))

        # print("拼接后的矩阵大小：", combined_features.shape)
        #将combined_features存储到brain_all_data_CCA文件夹下的data.csv文件中
        combined_features = pd.DataFrame(combined_features)
        combined_features.to_csv("/home/tangwenhao/TRC/data/brain_all_data_CCA/data.csv", index=False,header=False, mode='a')


exit()

#一下是示例代码
# 假设T1特征是 (116, 3) 矩阵，T2特征是 (116, 2) 矩阵
# 其中 116 表示ROI区域的数量，3和2表示各自提取的特征维度
features_T1 = np.random.rand(116, 3)  # T1特征矩阵
features_T2 = np.random.rand(116, 2)  # T2特征矩阵
# print(features_T1[1:10][:])
# print(features_T2[1:10][:])
# 1. 典型相关性分析 (CCA)
cca = CCA(n_components=min(features_T1.shape[1], features_T2.shape[1]))  # 提取最小维度的典型变量数量
T1_canonical, T2_canonical = cca.fit_transform(features_T1, features_T2)
print(T1_canonical.shape, T2_canonical.shape)
# 2. 按权重拼接典型相关性特征
# 可以选择前两个典型相关变量进行加权拼接
alpha = 0.5  # 调整权重参数
T1_weighted = alpha * T1_canonical
T2_weighted = (1 - alpha) * T2_canonical

# 拼接后的新特征矩阵 (116, 2) 大小
features_combined = T1_weighted + T2_weighted
# print(features_combined[1:10][:])
print("Combined Features Shape:", features_combined.shape)
