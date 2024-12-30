# KAGAN
code for KAGAN: Multimodal Classification of Alzheimer's Disease Based on Kolmogorov-Arnold Graph Attention Network

## Requirements

python >= 3.9.0

torch >= 1.10.0  

cuda >= 11.3  

Pytorch-Geometric

## Run the demo
```bash
bash git clone https://github.com/sheeprra/KAGAN.git
python data_peizhun.py
python data_pix_N4.py
python data_GetInf.py
python age_sex.py
python data_CCA.py
python kan.py
python combined_train.py
```



## Datasets
ADNI database: www.loni.ucla.edu/adni. 
You can specify a dataset as follows:
Visit the web site www.loni.ucla.edu/adni to download the ADNI dataset.

## Project name and introduction
 For the task of classifying Alzheimer's disease in a complex context, We propose a Multimodal Classification of Alzheimer's Disease Based on Kolmogorov-Arnold Graph Attention Network (KAGAN). The idea is to propose a new multimodal feature generation and fusion method that uses multilevel network interaction to improve classification accuracy.


