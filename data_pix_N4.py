""" No.1
该代码将所有图片进行处理，体素、方向、原点统一，然后进行N4BiasFieldCorrection校正
结果保存到brain_all_data_uni文件夹中
"""


import os
import SimpleITK as sitk
import time

# 假设TXT文档的路径和文件名是 known_files.txt 使用文档记录的形式使该代码可以中断后继续并多线程执行（开几个窗口就是几线程）
txt_path = '/home/tangwenhao/TRC/Brain/known_files.txt'
# 读取TXT文档中的文件名，并存储在集合中
known_files = set()
count = 0 
for path_all, dirs, files in os.walk('/home/tangwenhao/TRC/data/brain_all_data'):
    for file in files:
            if file.endswith('.nii.gz'):
                if os.path.exists(os.path.join(path_all.replace('brain_all_data','brain_all_data_uni'),file)):
                    count += 1
                    print(count)
                    continue
                with open(txt_path, 'r') as txt_file:
                    known_files = {line.strip() for line in txt_file}
                    if os.path.join(path_all,file) in known_files:
                        count += 1
                        print(count)
                        continue
                    else:
                        known_files.add(os.path.join(path_all,file))
                        # 如果需要，可以将更新后的集合写回TXT文档
                        with open(txt_path, 'w') as txt_file:
                                for known_file in known_files:
                                    txt_file.write(known_file + '\n')
                        time.sleep(1)
                        # 读取图像
                        image = sitk.ReadImage(os.path.join(path_all, file))
                        # 创建重采样过滤器
                        resample = sitk.ResampleImageFilter()
                        inputsize = image.GetSize()
                        inputspacing = image.GetSpacing()
                        outspacing = [1, 1, 1]
                        # 计算改变spacing后的size，用物理尺寸/体素的大小
                        outsize = [0, 0, 0]
                        outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
                        outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
                        outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

                        # 设置插值方法
                        resample.SetInterpolator(sitk.sitkLinear)  # 可以选择 sitk.sitkNearestNeighbor, sitk.sitkBSpline 等
                        resample.SetOutputPixelType(sitk.sitkFloat32)  # image用float32存

                        # 设置输出方向和原点，与原始图像一致
                        #(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
                        resample.SetOutputDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                        #(96,132.-78)
                        resample.SetOutputOrigin((96.0, 132.0, -78.0))
                        resample.SetOutputSpacing(outspacing)
                        resample.SetSize(outsize)

                        # 执行重采样
                        resampled_image = resample.Execute(image)

                        # # 应用高斯滤波
                        # sigma = 1.0  # 高斯核的标准差
                        # gaussian_filter = sitk.SmoothingRecursiveGaussian(image, sigma)
                        #使用N4BiasFieldCorrection校正MRI图像的偏置场,解决图像的亮度不均匀问题
                        resampled_image = sitk.N4BiasFieldCorrection(resampled_image, resampled_image > 0)
                        # # 保存重采样后的图像
                        # print(resampled_image.GetSize())
                        # print(resampled_image.GetSpacing())
                        # print(resampled_image.GetOrigin())
                        sitk.WriteImage(resampled_image, os.path.join(path_all.replace('brain_all_data','brain_all_data_uni'),file))
                        count += 1
                        print(count)

                        with open(txt_path, 'r') as txt_file:
                            known_files = {line.strip() for line in txt_file}

                        # 文件在集合中，从集合中删除
                        known_files.remove(os.path.join(path_all,file))
                        with open(txt_path, 'w') as txt_file:
                            for known_file in known_files:
                                txt_file.write(known_file + '\n')



#AAL变换
AAL = sitk.ReadImage("/home/tangwenhao/TRC/Brain/AAL.nii")
print(AAL.GetSize())
print(AAL.GetSpacing())
print(AAL.GetOrigin())
print(AAL.GetDirection())
resample = sitk.ResampleImageFilter()
inputsize = AAL.GetSize()
inputspacing = AAL.GetSpacing()
outspacing = [1, 1, 1]
# 计算改变spacing后的size，用物理尺寸/体素的大小
outsize = [0, 0, 0]
outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

# 设置插值方法
resample.SetInterpolator(sitk.sitkLinear)  # 可以选择 sitk.sitkNearestNeighbor, sitk.sitkBSpline 等
resample.SetOutputPixelType(sitk.sitkFloat32)  # image用float32存

# 设置输出方向和原点，与原始图像一致
#(-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
resample.SetOutputDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
#(96,132.-78)
resample.SetOutputOrigin((96.0, 132.0, -78.0))
resample.SetOutputSpacing(outspacing)
resample.SetSize(outsize)

# 执行重采样
resampled_image = resample.Execute(AAL)
print(resampled_image.GetSize())
print(resampled_image.GetSpacing())
print(resampled_image.GetOrigin())
print(resampled_image.GetDirection())
sitk.WriteImage(resampled_image,"/home/tangwenhao/TRC/Brain/AAL_resampled.nii")