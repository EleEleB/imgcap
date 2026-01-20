#!/usr/bin/env bash
#SBATCH --job-name=ic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=4:00:00
#SBATCH --output=ic_eval_%j.out
#SBATCH --error=ic_eval_%j.err

start=$(date +%s)

venv_ic/bin/python3 ic_eval4.py

end=$(date +%s)
runtime=$((end - start))
echo "Job runtime: $runtime seconds"
