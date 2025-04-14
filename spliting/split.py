#!/usr/bin/env python3
import os
import random
import shutil
import argparse

def split_dataset(images_dir, labels_dir, output_dir, train_ratio=0.7, test_ratio=0.2, val_ratio=0.1):
    """
    Divide los archivos .nii en las carpetas de imágenes y etiquetas
    en conjuntos de train, test y validation.
    """
    # Definir rutas de salida para imágenes
    imagesTr_dir = os.path.join(output_dir, "imagesTr")
    imagesTs_dir = os.path.join(output_dir, "imagesTs")
    imagesVal_dir = os.path.join(output_dir, "imagesVal")
    
    # Definir rutas de salida para etiquetas
    labelsTr_dir = os.path.join(output_dir, "labelsTr")
    labelsTs_dir = os.path.join(output_dir, "labelsTs")
    labelsVal_dir = os.path.join(output_dir, "labelsVal")
    
    # Crear todas las carpetas necesarias
    for folder in [imagesTr_dir, imagesTs_dir, imagesVal_dir,
                   labelsTr_dir, labelsTs_dir, labelsVal_dir]:
        os.makedirs(folder, exist_ok=True)
    
    # Obtener la lista de archivos .nii en el directorio de imágenes, 
    # ignorando archivos ocultos que comienzan con "._"
    files = [f for f in os.listdir(images_dir) 
             if f.endswith(".nii.gz") and not f.startswith("._")]
    random.shuffle(files)
    
    total_files = len(files)
    train_count = int(train_ratio * total_files)
    test_count = int(test_ratio * total_files)
    # El resto se asigna a validation
    val_count = total_files - train_count - test_count

    train_files = files[:train_count]
    test_files = files[train_count:train_count + test_count]
    val_files = files[train_count + test_count:]
    
    # Función para mover un archivo verificando si existe
    def move_file(file_name, src_dir, dst_dir):
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        if os.path.exists(src_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"⚠️  Archivo no encontrado: {src_path}")
    
    # Mover archivos de imágenes a sus respectivas carpetas
    for f in train_files:
        move_file(f, images_dir, imagesTr_dir)
    for f in test_files:
        move_file(f, images_dir, imagesTs_dir)
    for f in val_files:
        move_file(f, images_dir, imagesVal_dir)
    
    # Mover archivos de etiquetas (se asume que tienen el mismo nombre que las imágenes)
    for f in train_files:
        move_file(f, labels_dir, labelsTr_dir)
    for f in test_files:
        move_file(f, labels_dir, labelsTs_dir)
    for f in val_files:
        move_file(f, labels_dir, labelsVal_dir)
    
    print("✅ División completada:")
    print(f"  - Train: {len(train_files)} archivos")
    print(f"  - Test:  {len(test_files)} archivos")
    print(f"  - Val:   {len(val_files)} archivos")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script para dividir el dataset LiTS en train (70%), test (20%) y val (10%).")
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Directorio que contiene los archivos de imágenes (.nii)')
    parser.add_argument('--labels_dir', type=str, required=True,
                        help='Directorio que contiene los archivos de etiquetas (.nii)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directorio donde se creará el dataset dividido')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para la aleatorización (default: 42)')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    split_dataset(args.images_dir, args.labels_dir, args.output_dir)
