#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=8G      # Required memory per GPU (specify how many GB)
#SBATCH --time=2:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J image              # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file

# Execute the command
module load conda/2024.09
conda activate CS587

cd /home/shams3/CS-58700-Project/

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# cd /home/shams3/CS-58700-Project/baseline

# python gating_main.py -m multi_gated_c4 -e 50 -bs 64 -lr 1e-4
# python gating_main.py -m multi_gated_c4 -e 50 -bs 64 -lr 1e-4 --rotate_images
# python gating_main.py -m multi_gated_d4 -e 50 -bs 64 -lr 1e-4
# python gating_main.py -m multi_gated_d4 -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m multi_gated_c4_resnet -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_c4_resnet -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m multi_gated_d4_resnet -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4_resnet -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m multi_gated_steerable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_steerable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images 

# python Gating/gating_main.py -m multi_gated_c4_resnet  -d bird -e 50 -bs 64 -lr 1e-4
python Gating/gating_main.py -m multi_gated_c4_resnet  -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m multi_gated_d4_resnet  -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4_resnet  -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images