#!/bin/bash
#SBATCH --job-name=train_test_1.0
#SBATCH --time=24:00:00           # Tiempo máximo estimado para el entrenamiento
#SBATCH --mem=64G                 # Memoria RAM para la CPU (no afecta la GPU)
#SBATCH --cpus-per-task=8         # Número de CPUs a utilizar
#SBATCH --gres=gpu:1              # Solicita 1 GPU
#SBATCH --output=train_1_%j.out
#SBATCH --error=train_1_%j.err

#Ejecutar el script de entrenamiento
python train_1.py
