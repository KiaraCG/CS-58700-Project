#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=8G      # Required memory per GPU (specify how many GB)
#SBATCH --time=3:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J image              # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file

# Execute the command
module load conda/2024.09
conda activate CS587

# cd /home/shams3/CS-58700-Project/baseline

# python main.py -d inaturalist_mnist -m standard_cnn -e 100
# python main.py -d inaturalist_mnist -m standard_cnn -e 100 --rotate_images

# python main.py -d inaturalist_mnist -m d4cnn -e 100
# python main.py -d inaturalist_mnist -m d4cnn -e 100 --rotate_images

python main.py -d mnist_svhn -m standard_cnn -e 100
python main.py -d mnist_svhn -m standard_cnn -e 100 --rotate_images

python main.py -d mnist_svhn -m d4cnn -e 100
python main.py -d mnist_svhn -m d4cnn -e 100 --rotate_images