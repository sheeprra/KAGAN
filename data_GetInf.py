import os

import SimpleITK as sitk

from itertools import product
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops


import nibabel as nib
import numpy as np
import pandas as pd
import os

aal_path = '/Brain/AAL_C.nii.gz'
aal_img = nib.load(aal_path)
aal_data = aal_img.get_fdata()


list_AAL_num = [0.0,2001.0,2002.0,2101.0,2102.0, 
2111.0,2112.0,2201.0,2202.0,2211.0,2212.0,2301.0,2302.0,
2311.0,2312.0,2321.0,2322.0,2331.0,2332.0,2401.0,2402.0,
2501.0,2502.0,2601.0,2602.0,2611.0,2612.0,2701.0,2702.0,
3001.0,3002.0,4001.0,4002.0,4011.0,4012.0,4021.0,4022.0,
4101.0,4102.0,4111.0,4112.0,4201.0,4202.0,5001.0,5002.0,
5011.0,5012.0,5021.0,5022.0,5101.0,5102.0,5201.0,5202.0,
5301.0,5302.0,5401.0,5402.0,6001.0,6002.0,6101.0,6102.0,
6201.0,6202.0,6211.0,6212.0,6221.0,6222.0,6301.0,6302.0,
6401.0,6402.0,7001.0,7002.0,7011.0,7012.0,7021.0,7022.0,
7101.0,7102.0,8101.0,8102.0,8111.0,8112.0,8121.0,8122.0,
8201.0,8202.0,8211.0,8212.0,8301.0,8302.0,9001.0,9002.0,
9011.0,9012.0,9021.0,9022.0,9031.0,9032.0,9041.0,9042.0,
9051.0,9052.0,9061.0,9062.0,9071.0,9072.0,9081.0,9082.0,
9100.0,9110.0,9120.0,9130.0,9140.0,9150.0,9160.0,9170.0
]
list_AAL_num_array = np.array(list_AAL_num, dtype=np.float64)


def find_nearest_label(value):
    diff = np.abs(list_AAL_num_array - value)
    return list_AAL_num_array[np.argmin(diff)]


vectorized_find_nearest = np.vectorize(find_nearest_label,otypes=[np.float64])
aal_data_modified = vectorized_find_nearest(aal_data)

aal_labels = np.unique(aal_data_modified)


source_path = '/data/brain_all_data_PeiZhun/AD/T1'
for _,_,files in os.walk(source_path):
    for file in files:
        brain_mri_path = os.path.join(source_path,file)
        brain_mri_img = nib.load(brain_mri_path)
        brain_mri_data = brain_mri_img.get_fdata()
        i = int(file.split('.')[0])
        i = str(i)
        roi_info = []

        for label in aal_labels:
            if label == 0:  
                continue

            roi_mask = (aal_data_modified == label)

            min_val = brain_mri_data.min()
            max_val = brain_mri_data.max()

            brain_mri_data_normalized = ((brain_mri_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)

           
            levels = 256  
            glcm = np.zeros((levels, levels,1,3), dtype=int)  

      
            for x in range(brain_mri_data.shape[0] - 1):
                for y in range(brain_mri_data.shape[1] - 1):
                    for z in range(brain_mri_data.shape[2] - 1):
                        if roi_mask[x, y, z]: 
                            current_pixel_value = brain_mri_data_normalized[x, y, z]
                            
                         
                            if roi_mask[x + 1, y, z]: 
                                adjacent_pixel_value = brain_mri_data_normalized[x + 1, y, z]
                                glcm[current_pixel_value, adjacent_pixel_value,0,0] += 1

                            if roi_mask[x, y + 1, z]:
                                adjacent_pixel_value = brain_mri_data_normalized[x, y + 1, z]
                                glcm[current_pixel_value, adjacent_pixel_value,0,1] += 1

                            if roi_mask[x, y, z + 1]:
                                adjacent_pixel_value = brain_mri_data_normalized[x, y, z + 1]
                                glcm[current_pixel_value, adjacent_pixel_value,0,2] += 1

            contrast = graycoprops(glcm, 'contrast')
            dissimilarity = graycoprops(glcm, 'dissimilarity')
            homogeneity = graycoprops(glcm, 'homogeneity')
            energy = graycoprops(glcm, 'energy')
            correlation = graycoprops(glcm, 'correlation')
            ASM = graycoprops(glcm, 'ASM')
            entropy = -np.sum(glcm * np.log2(glcm + (glcm == 0)))  
            print(contrast, dissimilarity, homogeneity, energy, correlation, ASM, entropy)

            contrast_mean = np.mean(contrast)
            dissimilarity_mean = np.mean(dissimilarity)
            homogeneity_mean = np.mean(homogeneity)
            energy_mean = np.mean(energy)
            correlation_mean = np.mean(correlation)
            ASM_mean = np.mean(ASM)
            print(contrast_mean, dissimilarity_mean, homogeneity_mean, energy_mean, correlation_mean, ASM_mean, entropy)


            roi_data = brain_mri_data[roi_mask]
            mean_intensity = np.mean(roi_data)
            std_intensity = np.std(roi_data)

         
            roi_info.append({
                'ROI_Label': int(label),
                'Mean_Intensity': mean_intensity,
                'Std_Intensity': std_intensity,
                'Voxel_Count': np.sum(roi_mask), 
                'contrast_mean': contrast_mean,
                'dissimilarity_mean': dissimilarity_mean,
                'homogeneity_mean': homogeneity_mean,
                'energy_mean': energy_mean,
                'correlation_mean': correlation_mean,
                'ASM_mean': ASM_mean,
                'entropy': entropy

            })
            print(roi_info)

        roi_info_df = pd.DataFrame(roi_info)
       
        tar_path = '/data/brain_all_data_TeZhengTiQu/AD/T1'
        if not os.path.exists(os.path.join(tar_path,i)):
            os.makedirs(os.path.join(tar_path,i))
        roi_info_df.to_csv(os.path.join(tar_path,i,'t1_roi_info.csv'),  mode='a',header=False,index=False)

        
