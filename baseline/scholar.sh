#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=2G      # Required memory per GPU (specify how many GB)
#SBATCH --time=1:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J image              # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file

# Execute the command
# module load conda/2024.09
# conda activate CS587

# cd /home/shams3/CS-58700-Project/baseline



python main.py -m standard_cnn -d mnist -e 100
python main.py -m standard_cnn -d mnist -e 100 --rotate_images

#440784
# python main.py -m c4_invariant -d mnist -e 100 
#440785
# python main.py -m c4_invariant -d mnist -e 100 --rotate_images
# 440615
# python main.py -m c4_invariant -d colormnist -e 100
#440616
# python main.py -m c4_invariant -d colormnist -e 100 --rotate_images

#440617
# python main.py -m c4_invariant -d svhn -e 100
#440618
# python main.py -m c4_invariant -d svhn -e 100 --rotate_images


# python main.py -m c4_invariant -d inaturalist -e 100
# python main.py -m c4_invariant -d inaturalist -e 100 --rotate_images


#440619
# python main.py -m c4_equivariant -d mnist -e 100
#440620
# python main.py -m c4_equivariant -d mnist -e 100 --rotate_images
#440621
# python main.py -m c4_equivariant -d colormnist -e 100
#440622
# python main.py -m c4_equivariant -d colormnist -e 100 --rotate_images
#440623
# python main.py -m c4_equivariant -d svhn -e 100
#440624
# python main.py -m c4_equivariant -d svhn -e 100 --rotate_images


# python main.py -m c4_equivariant -d inaturalist -e 100