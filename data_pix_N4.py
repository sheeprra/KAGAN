import os
import SimpleITK as sitk
import time

txt_path = '/Brain/known_files.txt'

known_files = set()
count = 0 
for path_all, dirs, files in os.walk('/data/brain_all_data'):
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
      
                        with open(txt_path, 'w') as txt_file:
                                for known_file in known_files:
                                    txt_file.write(known_file + '\n')
                        time.sleep(1)
     
                        image = sitk.ReadImage(os.path.join(path_all, file))
     
                        resample = sitk.ResampleImageFilter()
                        inputsize = image.GetSize()
                        inputspacing = image.GetSpacing()
                        outspacing = [1, 1, 1]
               
                        outsize = [0, 0, 0]
                        outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
                        outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
                        outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])

             
                        resample.SetInterpolator(sitk.sitkLinear)  
                        resample.SetOutputPixelType(sitk.sitkFloat32)  

                       
                        resample.SetOutputDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
                 
                        resample.SetOutputOrigin((96.0, 132.0, -78.0))
                        resample.SetOutputSpacing(outspacing)
                        resample.SetSize(outsize)

                      
                        resampled_image = resample.Execute(image)

                     
                        resampled_image = sitk.N4BiasFieldCorrection(resampled_image, resampled_image > 0)
                      
                        sitk.WriteImage(resampled_image, os.path.join(path_all.replace('brain_all_data','brain_all_data_uni'),file))
                        count += 1
                        print(count)

                        with open(txt_path, 'r') as txt_file:
                            known_files = {line.strip() for line in txt_file}

                       
                        known_files.remove(os.path.join(path_all,file))
                        with open(txt_path, 'w') as txt_file:
                            for known_file in known_files:
                                txt_file.write(known_file + '\n')




AAL = sitk.ReadImage("/home/tangwenhao/TRC/Brain/AAL.nii")
print(AAL.GetSize())
print(AAL.GetSpacing())
print(AAL.GetOrigin())
print(AAL.GetDirection())
resample = sitk.ResampleImageFilter()
inputsize = AAL.GetSize()
inputspacing = AAL.GetSpacing()
outspacing = [1, 1, 1]

outsize = [0, 0, 0]
outsize[0] = round(inputsize[0] * inputspacing[0] / outspacing[0])
outsize[1] = round(inputsize[1] * inputspacing[1] / outspacing[1])
outsize[2] = round(inputsize[2] * inputspacing[2] / outspacing[2])


resample.SetInterpolator(sitk.sitkLinear)  
resample.SetOutputPixelType(sitk.sitkFloat32) 


resample.SetOutputDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))

resample.SetOutputOrigin((96.0, 132.0, -78.0))
resample.SetOutputSpacing(outspacing)
resample.SetSize(outsize)

resampled_image = resample.Execute(AAL)
print(resampled_image.GetSize())
print(resampled_image.GetSpacing())
print(resampled_image.GetOrigin())
print(resampled_image.GetDirection())
sitk.WriteImage(resampled_image,"/Brain/AAL_resampled.nii")
