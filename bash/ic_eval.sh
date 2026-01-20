#!/bin/bash
#SBATCH -J ic
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64g

start=$(date +%s)
mkdir -p .slurm
python src/ic_eval4.py

end=$(date +%s)
runtime=$((end - start))
echo "Job runtime: $runtime seconds"
