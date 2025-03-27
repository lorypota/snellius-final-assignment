# Snellius Final Assignment

This project contains several experiments that will be run on the Snellius supercomputer. It uses datasets downloaded from OpenML. The project is organized into a core library of shared utilities and per-task folders containing task-specific setups and experiments.

## File Structure

```
snellius-final-assignment/
├── core/
│   ├── __init__.py
│   ├── dataloading.py         # Loading utilities
│   ├── fewshot.py             # Few-shot helpers
│   ├── logger.py              # ModelLogger (optional)
│   └── utils.py               # Additional helpers
├── task1/
│   ├── base_setup.py          # Dataset loading, splits, transforms & loaders
│   ├── experiment1.py         # Model definition & training (W&B logging)
│   └── experiment2.py         # Alternative experiments...
├── task2/
│   ├── base_setup.py          # Similar setup for a different dataset
│   └── experiment1.py
├── requirements.txt           # Python package requirements
└── README.md
```

## Overview

- **Core Folder:**

  - **dataloading.py:** Implements `ImageDataset` for loading images and mapping labels.
  - **fewshot.py:** Provides classes and functions for few-shot learning experiments.
  - **logger.py:** Contains a `ModelLogger` class to log metrics, save models, and plot learning curves.
  - **utils.py:** Offers additional helper functions such as visualization.

- **Task Folders (e.g., task1):**
  - **base_setup.py:** Loads and processes the dataset (using OpenML), creates data splits, defines image transforms, and creates DataLoader objects. These variables are then imported into experiment scripts.
  - **experiment1.py (and experiment2.py, etc.):** These files define the model architecture (for example, a ResNet‑18 style `birdsCNN` in task1), set up the training procedure using PyTorch Lightning, and use WandB for logging.

## Running on Snellius

1. **Connect to Snellius:**

   ```bash
   ssh <username>@snellius.surf.nl
   ```

2. **Load Required Modules:**

   ```bash
   module purge #unload all currently loaded modules
   module load 2023
   module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
   ```

3. **(Optional) Create a Virtual Environment:**

   ```bash
   python -m venv my_venv
   source my_venv/bin/activate
   pip install --no-cache-dir -r requirements.txt
   ```

4. **Submit a Job (Example `job_task1.sh`):**

   ```bash
   #!/bin/bash
   #SBATCH --partition=gpu_mig
   #SBATCH --gpus=1
   #SBATCH --job-name=Task1_Exp1
   #SBATCH --ntasks=1
   #SBATCH --cpus-per-task=9
   #SBATCH --time=00:30:00
   #SBATCH --output=slurm_output_%A.txt
   #SBATCH --error=slurm_error_%A.txt

   module purge
   module load 2023
   module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1

   source my_venv/bin/activate  # if using a virtual environment

   srun python task1/experiment1.py
   ```

   Then:

   ```bash
   sbatch job_task1.sh
   squeue -u <your_username>
   scancel <job_id>
   ```

## Additional Useful Commands

- **Data Transfer:**

  ```bash
  scp -r /local/path/to/snellius-final-assignment <username>@snellius.surf.nl:$HOME/
  ```

- **Module Commands:**

  ```bash
  module avail
  module load <module_name>
  module list
  module unload <module_name>
  module purge
  module whatis <module_name>
  ```

- **Job Management:**
  ```bash
  sbatch <job_script.sh>
  squeue -u <your_username>
  scontrol show job <job_id>
  scancel <job_id>
  ```

## Further Resources

- [Snellius OnDemand](https://ondemand.snellius.surf.nl/)
- [SURF Wiki on Snellius](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius)
- [SLURM sbatch Documentation](https://slurm.schedmd.com/sbatch.html)
