#!/bin/bash
#SBATCH -N 1
#SBATCH -n 64
#SBATCH -p xahcnormal
config_name=$*
python3 ./main.py ${config_name}
