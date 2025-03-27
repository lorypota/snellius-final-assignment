#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=Task1_Exp1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:24:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source ../my_venv/bin/activate

# Set PYTHONPATH to the project root (one level above the current folder)
export PYTHONPATH=$(pwd)/..

# Run experiment1 as a module using its full package name.
srun python -u -m task1_birds.experiment1