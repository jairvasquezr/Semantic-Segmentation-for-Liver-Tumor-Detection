#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert NIfTI images (.nii.gz) to PNG by applying windowing to the images
"""

import numpy as np
import nibabel as nib
import os
import imageio
from tqdm import tqdm

def safe_write_image(output_path, image, retries=1):
    for attempt in range(retries):
        imageio.imwrite(output_path, image)
        if os.path.getsize(output_path) > 0:
            return True
        else:
            print(f"Reintentando escribir {output_path} (intento {attempt+1}/{retries})")
    return False

def apply_windowing(img_array, window_center, window_width):
    min_hu = window_center - (window_width / 2)
    max_hu = window_center + (window_width / 2)
    img_array = np.clip(img_array, min_hu, max_hu)
    img_array = (img_array - min_hu) / (max_hu - min_hu) * 255
    return img_array.astype(np.uint8)

def convert_nii_to_png_with_windowing(input_dir, output_dir, window_center=40, window_width=400):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir), desc=f"Processing {input_dir}"):
        if filename.startswith("._"):
            continue
        if filename.endswith(".nii.gz"):
            nii_path = os.path.join(input_dir, filename)
            try:
                img = nib.load(nii_path)
                img_array = img.get_fdata()
                img_array = apply_windowing(img_array, window_center, window_width)
                for i in range(img_array.shape[2]):
                    slice_img = img_array[:, :, i]
                    output_filename = f"{filename.replace('.nii.gz', '')}_slice{i}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    success = safe_write_image(output_path, slice_img, retries=1)
                    if not success:
                        print(f"No se pudo guardar correctamente {output_path}.")
            except Exception as e:
                print(f"Error processing {nii_path}: {e}")

def convert_nii_to_png(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in tqdm(os.listdir(input_dir), desc="Convirtiendo máscaras NIfTI a PNG"):
        if filename.startswith("._"):
            continue
        if filename.endswith(".nii.gz"):
            nii_path = os.path.join(input_dir, filename)
            try:
                img = nib.load(nii_path)
                img_array = img.get_fdata().astype(np.uint8)
                for i in range(img_array.shape[2]):
                    slice_img = img_array[:, :, i]
                    output_filename = f"{filename.replace('.nii.gz', '')}_slice{i}.png"
                    output_path = os.path.join(output_dir, output_filename)
                    success = safe_write_image(output_path, slice_img, retries=1)
                    if not success:
                        print(f"No se pudo guardar correctamente {output_path}.")
            except Exception as e:
                print(f"Error procesando {nii_path}: {e}")

if __name__ == "__main__":
    # Ajusta las rutas según tu estructura en Compute Canada
    input_images_train = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/imagesTr"
    output_images_train = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/imagesTr_png"
    input_images_val   = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/imagesVal"
    output_images_val  = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/imagesVal_png"
    input_images_ts = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/imagesTs"
    output_images_ts = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/imagesTs_png"

    input_labels_Tr = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/labelsTr"
    output_labels_Tr = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/labelsTr_png"
    input_labels_Val = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/labelsVal"
    output_labels_Val = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/labelsVal_png"
    input_labels_Ts = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited/labelsTs"
    output_labels_Ts = "/home/jair/projects/def-saadi/jair/Data/data_original/data_splited_png/labelsTs_png"


    #print("Procesando imágenes de entrenamiento...")
    #convert_nii_to_png_with_windowing(input_images_train, output_images_train, window_center=40, window_width=400)

    #print("Procesando imágenes de validación...")
    #convert_nii_to_png_with_windowing(input_images_val, output_images_val, window_center=40, window_width=400)

    print("Procesando imágenes de test...")
    convert_nii_to_png_with_windowing(input_images_ts, output_images_ts, window_center=40, window_width=400)


    print("Procesando etiquetas de entrenamiento...")
    convert_nii_to_png(input_labels_Tr, output_labels_Tr)

    print("Procesando etiquetas de validación...")
    convert_nii_to_png(input_labels_Val, output_labels_Val)

    print("Procesando etiquetas de test..")
    convert_nii_to_png(input_labels_Ts, output_labels_Ts)


    print("Processing complete")
