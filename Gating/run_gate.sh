#!/bin/bash
# FILENAME:  scholar

#SBATCH --export=ALL          # Export your current environment settings to the job environment
#SBATCH -A gpu                # Account name
#SBATCH --ntasks=1            # Number of MPI ranks per node (one rank per GPU)
#SBATCH --cpus-per-task=4     # Number of CPU cores per MPI rank (change this if needed)
#SBATCH --gres=gpu:1          # Use one GPU
#SBATCH --mem-per-cpu=8G      # Required memory per GPU (specify how many GB)
#SBATCH --time=4:00:00        # Total run time limit (hh:mm:ss)
#SBATCH -J image              # Job name
#SBATCH -o slurm_logs/%j      # Name of stdout output file

# Execute the command
module load conda/2024.09
conda activate CS587

cd /home/shams3/CS-58700-Project/

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)

# cd /home/shams3/CS-58700-Project/baseline

# python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images 

# python -c "print('bird with resnet learnable gates')"

# python Gating/gating_main.py -m multi_gated_c4_resnet  --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_c4_resnet  --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m multi_gated_d4_resnet  --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4_resnet  --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images


# python -c "print('svhn steerable with learnable gates and at the end is mutli d4 rotate and reflect images')"

python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images 
# python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images

python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images 
python Gating/gating_main.py -m multi_gated_steerable --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images


# python -c "print('bird with frozen and learnable gates')"
# python Gating/gating_main.py -m c4_tsbn  --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m c4_tsbn  --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m c4_tsbn  --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-3 --rotate_images --reflect_images

# python -c "print('svhn d4 learnable')"

# python Gating/gating_main.py -m multi_gated_d4_resnet --gate_cls none -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4_resnet --gate_cls none -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images

# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images    
# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images




python Gating/gating_main.py -m multi_gated_c4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
python Gating/gating_main.py -m multi_gated_c4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images
python Gating/gating_main.py -m multi_gated_c4 --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images

# python Gating/gating_main.py -m multi_gated_d4_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m multi_gated_d4_resnet --gate_cls learnable -d svhn -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images

# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m plain_resnet --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images

# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4
# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images
# python Gating/gating_main.py -m multi_gated_d4 --gate_cls learnable -d bird -e 50 -bs 64 -lr 1e-4 --rotate_images --reflect_images
