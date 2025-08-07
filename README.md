# Semantic-Segmentation-for-Liver-Tumor-Detection
This project applies the SegFormer model for semantic segmentation of liver and tumor regions in CT scans, using the LiTS (Liver Tumor Segmentation) dataset. It includes dataset preprocessing, training, and inference pipelines, along with reproducible code and pretrained checkpoints.

## Problem Statement
The liver is one of the largest organs in the human body. Liver cancer, a condition in which malignant (cancerous) cells form in liver tissue, poses a serious global health concern. It ranks among the top three causes of cancer-related deaths in many countries. Alarmingly, the liver cancer mortality rate is projected to rise by 56.4% by the year 2040.

Computed Tomography (CT) remains the standard imaging technique for detecting liver tumors. However, manual segmentation of the liver and tumors in CT scans is a time-consuming and subjective task, highlighting the need for automated and reliable solutions.

## Objective
This work aims to optimize the radiotherapy planning process by implementing a Deep Learning model—SegFormer, based on Vision Transformers—to automate the segmentation of the liver and tumors in CT images. The goal is to enhance segmentation accuracy while significantly reducing the need for manual intervention.

## Dataset - LiTS (Liver Tumor Segmentation Challenge)

Download the dataset from the official site[LiTS challenge](http://medicaldecathlon.com/) or from Kaggle (https://www.kaggle.com/datasets/andrewmvd/lits-png).  
Preprocessing includes:

## Preprocessing
In the NIFTI format, pixel intensities are represented in Hounsfield Units (HU), which measure tissue density in CT images. To enhance contrast between the liver, tumors and surrounding structures, we apply windowing with a center of 40 HU and a width of 400 HU, restricting the intensity range to [-160,240] HU. Voxels with values outside this range are clipped accordingly: those below -160 HU are set to -160 HU, while those above 240 HU are capped at 240 HU. In addition, the images are normalized to the [0,255] range to standardize the pixel intensities for further processing. Finally, images whose corresponding masks lacked expert annotations were discarded, as they did not provide relevant information for the study. This step also helped mitigate the class imbalance problem.
- Conversion from `.nii` to `.png`
- HU windowing and normalization
- Mask generation for liver and tumor labels

## Expertiment
The proposed method was implemented using Python and the Pytorch deep learning library. The experiments were conducted on the Cedar cluster, which consists of a variety of nodes, including large-memory nodes and GPU-accelerated nodes. The training process utilized 12.18 GB of RAM on an Intel E5-2650 v4 Broadwell CPU 2.2GHz and an NVIDIA P100 Pascal GPU with 12 GB of memory.

## Results
A SegFormer-based model was proposed for the automatic segmentation of the liver and tumors in CT images. The model was validated on the LiTS dataset, achieving Dice scores of 0.949 for liver segmentation and 0.775 for tumor segmentation—outperforming the nnU-Net benchmark from MICCAI 2018 in tumor delineation.

This approach delivers accurate segmentation results while significantly reducing manual workload, saving time, and enabling earlier clinical intervention.
![segmentation_1](https://github.com/user-attachments/assets/63482cd2-74e2-4f22-960f-0a051c0af990)

![probability_map](https://github.com/user-attachments/assets/3aa7a04d-865f-4695-adba-2e33f00dd260)

Table 1 summarizes the liver segmentation results in terms of the Dice coefficient.

<img width="504" height="206" alt="image" src="https://github.com/user-attachments/assets/490c91c3-4103-4496-90e7-d5a002d069a1" />

Table 2 presents the performance of various architectures for tumor segmentation. It can be observed that our model outperforms the method proposed by F. Isensee et al. (nnU-Net).

<img width="551" height="306" alt="image" src="https://github.com/user-attachments/assets/578a00c6-6fdf-4c33-87b1-f6a2dae52137" />
