
import numpy as np
import SimpleITK as sitk

def standardize_image(image):
    image_array = sitk.GetArrayFromImage(image)
    standardized_array = (image_array - np.mean(image_array)) / np.std(image_array)
    standardized_image = sitk.GetImageFromArray(standardized_array)
    standardized_image.CopyInformation(image)
    return standardized_image

def normalize_image_to_range(image, output_min, output_max):
    image_array = sitk.GetArrayFromImage(image)
    input_min = image_array.min()
    input_max = image_array.max()
    normalized_array = (image_array - input_min) * (output_max - output_min) / (input_max - input_min) + output_min
    normalized_image = sitk.GetImageFromArray(normalized_array)
    normalized_image.CopyInformation(image)
    return normalized_image
            

mni_template1 = sitk.ReadImage("/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t1_tal_nlin_sym_09c.nii")
mni_template2 = sitk.ReadImage("/data/mni_icbm152_nlin_sym_09c/mni_icbm152_t2_tal_nlin_sym_09c.nii")


moving_image = sitk.ReadImage('data/brain_all_data_uni/AD/0/T1/Sag_Accel_IR-FSPGR_20181016113556.nii.gz')


moving_image_standardized = standardize_image(moving_image)
resampled_image = normalize_image_to_range(moving_image_standardized, 0, 100)



registration_method = sitk.ImageRegistrationMethod()

registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
registration_method.SetMetricSamplingPercentage(0.01)

registration_method.SetInterpolator(sitk.sitkLinear)

registration_method.SetOptimizerAsGradientDescent(learningRate=0.05, numberOfIterations=300, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

initial_transform = sitk.CenteredTransformInitializer(mni_template1,
                                                      resampled_image,
                                                      sitk.Euler3DTransform(),
                                                      sitk.CenteredTransformInitializerFilter.GEOMETRY)
registration_method.SetInitialTransform(initial_transform, inPlace=False)

final_transform = registration_method.Execute(sitk.Cast(mni_template1, sitk.sitkFloat32),
                                              sitk.Cast(resampled_image, sitk.sitkFloat32))

moving_resampled = sitk.Resample(resampled_image, mni_template1, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

print('Final metric value: {0}'.format(registration_method.GetMetricValue()))
print('Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

sitk.WriteImage(moving_resampled, 'registered_image_path.nii.gz')
