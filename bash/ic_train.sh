#!/usr/bin/env bash
#SBATCH --job-name=ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --output=./.slurm/%A/%a_output.log
#SBATCH --error=./.slurm/%A/%a_error.log

start=$(date +%s)

.env/bin/python3 src/ic_train4.py

end=$(date +%s)
runtime=$((end - start))
echo "Job runtime: $runtime seconds"
