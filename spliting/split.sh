#!/bin/bash
#SBATCH --job-name=split_lits
#SBATCH --nodes=1
#SBATCH --time=20:00       # Ajusta el tiempo según sea necesario
#SBATCH --mem=16G             # Memoria total del nodo
#SBATCH --cpus-per-task=4     # Número de núcleos
#SBATCH --output=split_lits_%j.out
#SBATCH --error=split_lits_%j.err

# Cargar módulos necesarios
module load python/3.11

# Activar el entorno virtual
source ~/jupyter_py3/bin/activate

# Definir rutas
IMAGES_DIR="/home/jair/projects/def-saadi/jair/Data/data_original/images"
LABELS_DIR="/home/jair/projects/def-saadi/jair/Data/data_original/labels"
OUTPUT_DIR="/home/jair/projects/def-saadi/jair/Data/data_original/data_splited"

# Ejecutar el script de división
python split.py --images_dir "$IMAGES_DIR" --labels_dir "$LABELS_DIR" --output_dir "$OUTPUT_DIR"
