#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=48
#SBATCH --time=48:00:00
#SBATCH --job-name=e_chepuri
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
#SBATCH --partition=standard


module load DL-CondaPy/3.9

cd /home/pushkal.ee.iith/Brittney_Temperature

python3 Eldar_Chepuri.py