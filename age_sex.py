#这里统计年龄与性别
import pandas as pd
import numpy as np
#读取all.csv文件中的数据
data = pd.read_csv('/home/tangwenhao/TRC/Brain/all.csv')
data_all = pd.read_csv('/home/tangwenhao/TRC/Brain/data_ADNI.csv')
#输出数据的前5行
age_CN = []
age_AD = []
age_MCI = []
sex_CN_male = 0
sex_CN_female = 0
sex_AD_male = 0
sex_AD_female = 0
sex_MCI_male = 0
sex_MCI_female = 0
count = 0
for i in range(len(data)):
    if data['group'][i] == 'CN':
        #读取data_all中imageUID等于data['imageUID_T2'][i]的记录中的年龄
        age = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['subjectAge'].values
        age_CN.append(age)
        sex = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['sex'].values
        if sex =='F':
            sex_CN_female += 1
        else:
            sex_CN_male += 1
    elif data['group'][i] == 'AD':
        age = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['subjectAge'].values
        age_AD.append(age)
        sex = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['sex'].values
        if sex =='F':
            sex_AD_female += 1
        else:
            sex_AD_male += 1
    else:
        age = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['subjectAge'].values
        age_MCI.append(age)
        sex = data_all[data_all['imageUID'] == data['imageUID_T2'][i]]['sex'].values
        if sex =='F':
            sex_MCI_female += 1
        else:
            sex_MCI_male += 1
#计算age_CN的均值与标准差
age_CN = np.array(age_CN)
age_CN = age_CN.flatten()
mean_age_CN = np.mean(age_CN)
std_age_CN = np.std(age_CN)
print(mean_age_CN,std_age_CN)

age_AD = np.array(age_AD)
age_AD = age_AD.flatten()
mean_age_AD = np.mean(age_AD)
std_age_AD = np.std(age_AD)
print(mean_age_AD,std_age_AD)

age_MCI = np.array(age_MCI)
age_MCI = age_MCI.flatten()
mean_age_MCI = np.mean(age_MCI)
std_age_MCI = np.std(age_MCI)
print(mean_age_MCI,std_age_MCI)

print(sex_CN_male,sex_CN_female,sex_AD_male,sex_AD_female,sex_MCI_male,sex_MCI_female)