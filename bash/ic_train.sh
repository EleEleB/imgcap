#!/bin/bash
#SBATCH --job-name=ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%J_output.log
#SBATCH --error=./.slurm/%J_error.log

start=$(date +%s)
python3 src/ic_train4.py

end=$(date +%s)
runtime=$((end - start))
echo "Job runtime: $runtime seconds"
