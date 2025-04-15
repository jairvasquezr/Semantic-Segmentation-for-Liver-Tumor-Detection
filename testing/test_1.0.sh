#!/bin/bash
#SBATCH --job-name=test_1.0
#SBATCH --time=02:00:00           # Tiempo máximo estimado
#SBATCH --mem=32G                 # Memoria RAM para la CPU (no afecta la GPU)
#SBATCH --cpus-per-task=8         # Número de CPUs a utilizar
#SBATCH --output=test_1.0_%j.out
#SBATCH --error=test_1.0_%j.err

# Cargar los módulos necesarios
module load python/3.11
module load cuda/12.2
module load gcc arrow/19.0.1

# Activar el entorno virtual
source ~/jupyter_py3/bin/activate

#Ejecutar el script de entrenamiento
python test_1.0.py
