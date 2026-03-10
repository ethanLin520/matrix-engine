#!/bin/bash
#
#SBATCH --mail-user=lin12@cs.uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --job-name=matrix_engine
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=900
#SBATCH --exclusive
#SBATCH --time=10:00
#SBATCH --output=slurm/out/%j.%N.stdout
#SBATCH --error=slurm/out/%j.%N.stderr

mkdir -p slurm/out

