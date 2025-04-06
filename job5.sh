#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=task5_exp1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:00:00
#SBATCH --output=slurm_output_%A.txt
#SBATCH --error=slurm_error_%A.txt

module purge
module load 2023
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

source my_venv/bin/activate

# Set PYTHONPATH to the project root (one level above the current folder)
export PYTHONPATH=$(pwd)

# Create results directory in the project root
mkdir -p results
mkdir -p results/checkpoints
mkdir -p results/outputs

# Run experiment5.py with OpenML API key
srun python -u -m task5_openml.experiment1.py --api_key YOUR_API_KEY --epochs 25 --batch_size 64