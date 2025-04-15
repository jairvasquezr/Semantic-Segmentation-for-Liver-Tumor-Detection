#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from sklearn.metrics import accuracy_score, recall_score, precision_score
from tqdm import tqdm

# ============================
# 1. Definir las rutas para test (actualiza según corresponda)
# ============================
test_img_dir = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/imagesTs_red_png"
test_mask_dir = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/labelsTs_red_png"
model_save_path = "/home/jair/projects/def-saadi/jair/models/model_test_1.0"

# ============================
# 2. Funciones auxiliares para carga de datos
# ============================
def load_image(path):
    return Image.open(path).convert("RGB")

def load_mask(path):
    return Image.open(path).convert("L")

class LiverTestDataset(Dataset):
    def __init__(self, img_dir, mask_dir, feature_extractor):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.feature_extractor = feature_extractor
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])

        image = load_image(img_path)
        mask  = load_mask(mask_path)

        image = np.array(image)
        encoding = self.feature_extractor(images=image, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze(0)

        mask = np.array(mask, dtype=np.int64)
        mask = torch.tensor(mask, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": mask}

# ============================
# 3. Cargar modelo y feature extractor
# ============================
print("Cargando modelo y feature extractor...")
feature_extractor = SegformerImageProcessor.from_pretrained(model_save_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_save_path)
model.eval()  # Modo evaluación
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ============================
# 4. Crear el dataset y dataloader de test
# ============================
test_dataset = LiverTestDataset(test_img_dir, test_mask_dir, feature_extractor)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# ============================
# 5. Evaluación del modelo
# ============================
all_predictions = []
all_labels = []

print("Evaluando el modelo...")
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Procesando batches"):
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        # Obtener las predicciones del modelo
        outputs = model(pixel_values=pixel_values).logits  # forma: (batch, num_labels, H, W)
        preds = torch.argmax(outputs, dim=1)

        # Ajuste de tamaño si es necesario para que preds y labels tengan la misma forma
        if preds.shape != labels.shape:
            labels = labels.unsqueeze(1).float()
            labels = F.interpolate(labels, size=preds.shape[1:], mode='nearest').squeeze(1).long()

        all_predictions.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenar todos los resultados
all_predictions = np.concatenate(all_predictions, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Aplanar las matrices para calcular las métricas a nivel de pixel
flat_preds = all_predictions.flatten()
flat_labels = all_labels.flatten()

# ============================
# 6. Calcular las métricas globales
# ============================
# Overall Accuracy
overall_acc = accuracy_score(flat_labels, flat_preds)
print(f"\nOverall Accuracy: {overall_acc:.4f}")

# Recall (macro): Es el promedio de recall calculado por cada clase. Indica qué tan bien el modelo es capaz de recuperar cada clase por separado.
recall_macro = recall_score(flat_labels, flat_preds, average='macro', zero_division=0)
print(f"Recall (macro): {recall_macro:.4f}")

# Precision (macro)
precision_macro = precision_score(flat_labels, flat_preds, average='macro', zero_division=0)
print(f"Precision (macro): {precision_macro:.4f}")

# ============================
# 7. Calcular métricas por clase
# ============================
num_classes = 3

# IoU por clase
def compute_iou_per_class(preds, labels, num_classes):
    ious = []
    for cls in range(num_classes):
        TP = np.sum((preds == cls) & (labels == cls))
        FP = np.sum((preds == cls) & (labels != cls))
        FN = np.sum((preds != cls) & (labels == cls))
        iou = TP / (TP + FP + FN + 1e-7)
        ious.append(iou)
    return ious

ious = compute_iou_per_class(flat_preds, flat_labels, num_classes)
mean_iou = np.mean(ious)

# Accuracy por clase (se calcula como la proporción de píxeles correctamente clasificados en cada clase)
per_class_acc = []
for cls in range(num_classes):
    true_cls = (flat_labels == cls)
    correct = np.sum((flat_labels == cls) & (flat_preds == cls))
    total = np.sum(true_cls)
    acc_cls = correct / total if total > 0 else 0.0
    per_class_acc.append(acc_cls)

# Recall por clase
per_class_recall = recall_score(flat_labels, flat_preds, average=None, zero_division=0)

# Precision por clase
per_class_precision = precision_score(flat_labels, flat_preds, average=None, zero_division=0)

# Mostrar resultados por clase
print("\nMétricas por clase:")
for cls in range(num_classes):
    print(f"\nClase {cls}:")
    print(f"  IoU: {ious[cls]:.4f}")
    print(f"  Accuracy: {per_class_acc[cls]:.4f}")
    print(f"  Recall: {per_class_recall[cls]:.4f}")
    print(f"  Precision: {per_class_precision[cls]:.4f}")

print("\nMétricas Promedio:")
print(f"  Mean IoU: {mean_iou:.4f}")
print(f"  Mean Accuracy: {np.mean(per_class_acc):.4f}")
print(f"  Recall (macro): {recall_macro:.4f}")
print(f"  Precision (macro): {precision_macro:.4f}")
