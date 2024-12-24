""" No.2
该模块对数据进行配准，过程已在ITK-SNAP中完成，此处仅为代码示例
结果保存在brain_all_data_PeiZhun文件夹中
实际上在文件夹中保存的是手动配准过后的数据
"""

import numpy as np
import SimpleITK as sitk
# 0 1归一化
def standardize_image(image):
    image_array = sitk.GetArrayFromImage(image)
    standardized_array = (image_array - np.mean(image_array)) / np.std(image_array)
    standardized_image = sitk.GetImageFromArray(standardized_array)
    standardized_image.CopyInformation(image)
    return standardized_image
# 0-100 归一化
def normalize_image_to_range(image, output_min, output_max):
    image_array = sitk.GetArrayFromImage(image)
    input_min = image_array.min()
    input_max = image_array.max()
    normalized_array = (image_array - input_min) * (output_max - output_min) / (input_max - input_min) + output_min
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image
            

mni_template1 = sitk.ReadImage("/home/tangwenhao/TRC/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii")
mni_template2 = sitk.ReadImage("/home/tangwenhao/TRC/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t2_tal_nlin_sym_09c.nii")


moving_image = sitk.ReadImage('/home/tangwenhao/TRC/data/brain_all_data_uni/AD/0/T1/Sag_Accel_IR-FSPGR_20181016113556.nii.gz')
#moving_image = sitk.ReadImage=("/home/tangwenhao/TRC/data/brain_all_data/SMC/118/T2/Sagittal_3D_FLAIR_20180123145800.nii.gz")
# resampled_image_array = sitk.GetArrayFromImage(mni_template1)
# resampled_image_array = resampled_image_array.transpose(2, 1, 0)

moving_image_standardized = standardize_image(moving_image)
resampled_image = normalize_image_to_range(moving_image_standardized, 0, 100)


# 初始化配准方法
registration_method = sitk.ImageRegistrationMethod()
# 设置相似性度量标准 
registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) #【互信息方法】
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)#【取样条件】
registration_method.SetMetricSamplingPercentage(0.01)
# 设置插值方法
registration_method.SetInterpolator(sitk.sitkLinear)
#【优化器参数配置： 步长学习率，迭代次数，收敛验证的窗宽】
registration_method.SetOptimizerAsGradientDescent(learningRate=0.05, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
# 使用多分辨率框架
# registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[8, 4, 2, 1])
# registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[3, 2, 1, 0])
# 初始变换设置
initial_transform = sitk.CenteredTransformInitializer(mni_template1,
                                                      resampled_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
registration_method.SetInitialTransform(initial_transform, inPlace=False)
# 执行配准
final_transform = registration_method.Execute(sitk.Cast(mni_template1, sitk.sitkFloat32),
                                              sitk.Cast(resampled_image, sitk.sitkFloat32))
# 将最终变换应用到移动图像
moving_resampled = sitk.Resample(resampled_image, mni_template1, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())
# 输出最终的度量值和优化器的停止条件
print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
# 保存结果
sitk.WriteImage(moving_resampled, 'registered_image_path.nii.gz')
