#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import nibabel as nb
import imageio
import time
import gc
import torch
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from evaluate import load
import torch.nn.functional as F
from transformers import (SegformerImageProcessor, SegformerForSemanticSegmentation, 
                          TrainingArguments, Trainer, EarlyStoppingCallback)

# ============================
# 1. Definir las rutas (actualiza estas rutas para Compute Canada)
# ============================
train_img_dir = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/imagesTr_red_png"
train_mask_dir = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/labelsTr_red_png"
val_img_dir   = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/imagesVal_red_png"
val_mask_dir  = "/home/jair/projects/def-saadi/jair/Data/data_original/data_reduced_png/labelsVal_red_png"
# Directorio para guardar el modelo final
model_save_path = "/home/jair/projects/def-saadi/jair/models/model_test_1.0"

# ============================
# 2. Funciones auxiliares para carga de datos
# ============================
def load_image(path):
    return Image.open(path).convert("RGB")

def load_mask(path):
    return Image.open(path).convert("L")

class LiverDataset(Dataset):
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
        # Reasignar valores en la máscara según tus necesidades:
	#mask[mask == 127] = 1
        #mask[mask == 255] = 2
        mask = torch.tensor(mask, dtype=torch.long)

        return {"pixel_values": pixel_values, "labels": mask}

# ============================
# 3. Configuración del modelo y datasets
# ============================
model_checkpoint = "nvidia/segformer-b2-finetuned-ade-512-512"
feature_extractor = SegformerImageProcessor.from_pretrained(model_checkpoint)

train_dataset = LiverDataset(train_img_dir, train_mask_dir, feature_extractor)
val_dataset   = LiverDataset(val_img_dir, val_mask_dir, feature_extractor)

# ============================
# 4. Definir los argumentos de entrenamiento
# ============================
training_args = TrainingArguments(
    output_dir="/home/jair/projects/def-saadi/jair/models/model_test_1.0_results",
    run_name="model_test_1.0",
    # "eval_strategy" puede ser "epoch" o "steps":
    #  - "epoch": Evalúa al finalizar cada época.
    #  - "steps": Evalúa cada cierto número de pasos (definir con eval_steps), útil cuando las épocas son muy largas.
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    eval_accumulation_steps=8,
    num_train_epochs=30,
    weight_decay=0.01,
    push_to_hub=False,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    report_to=[],                   # Deshabilita WandB
)

# ============================
# 5. Configurar el modelo
# ============================
model = SegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=3,
    ignore_mismatched_sizes=True
)

# Liberar memoria innecesaria
gc.collect()
torch.cuda.empty_cache()

# ============================
# 6. Definir la métrica y función de evaluación
# ============================
metric = load("mean_iou")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.from_numpy(logits), dim=1)

    # Ajustar tamaño de labels si es necesario
    if predictions.shape != labels.shape:
        labels_tensor = torch.from_numpy(labels).unsqueeze(1).float()
        labels_resized = F.interpolate(
            labels_tensor, size=predictions.shape[1:], mode='nearest'
        ).squeeze(1)
        labels = labels_resized.numpy()
        predictions = predictions.numpy()
    else:
        predictions = predictions.numpy()

    results = metric.compute(
        predictions=predictions,
        references=labels,
        num_labels=3,
        ignore_index=-100
    )

    final_results = {}
    for key, value in results.items():
        final_results[key] = value.tolist() if isinstance(value, np.ndarray) else value

    return final_results

# ============================
# 7. Configurar el Trainer
# ============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda batch: {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    },
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# ============================
# 8. Entrenamiento y guardado del modelo
# ============================
if __name__ == '__main__':
    trainer.train()
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    feature_extractor.save_pretrained(model_save_path)
