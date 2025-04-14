# Semantic-Segmentation-for-Liver-Tumor-Detection
This project applies the SegFormer model for semantic segmentation of liver and tumor regions in CT scans, using the LiTS (Liver Tumor Segmentation) dataset. It includes dataset preprocessing, training, and inference pipelines, along with reproducible code and pretrained checkpoints.


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
