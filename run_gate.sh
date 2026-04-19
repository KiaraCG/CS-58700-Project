#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=8G      # Required memory per GPU (specify how many GB)
#SBATCH --time=1:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J image              # Job name
#SBATCH -o Gating/slurm_logs/%j      # Name of stdout output file

# Execute the command
# module load conda/2024.09
# conda activate CS587

# cd /home/shams3/CS-58700-Project/baseline

# MultiGated C4
python gating_main.py -m multi_gated_c4 -d inaturalist_mnist -e 30 -bs 64 -lr 1e-4
python gating_main.py -m multi_gated_c4 -d inaturalist_mnist -e 30 -bs 64 -lr 1e-4 --rotate_images

# MultiGated D4
python gating_main.py -m multi_gated_d4 -d inaturalist_mnist -e 30 -bs 64 -lr 1e-4
python gating_main.py -m multi_gated_d4 -d inaturalist_mnist -e 30 -bs 64 -lr 1e-4 --rotate_images

