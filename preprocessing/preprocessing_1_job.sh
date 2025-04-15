#!/bin/bash
#SBATCH --job-name=preprocessing_data
#SBATCH --time=01:00:00         # Ajusta el tiempo máximo según lo requiera tu trabajo
#SBATCH --mem=32G               # Memoria requerida
#SBATCH --cpus-per-task=4       # Número de CPUs
#SBATCH --output=preprocess_dataoriginal_%j.out
#SBATCH --error=preprocess_dataoriginal_%j.err

# Cargar el módulo de Python 
module load python/3.11

# Activar el entorno virtual
source ~/jupyter_py3/bin/activate

# Ejecutar el script Python con la ruta completa
python preprocessing_1.py
