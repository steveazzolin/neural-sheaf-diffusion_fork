#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=nsd_test
#SBATCH -t 0-06:00
#SBATCH --output=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/test.txt
#SBATCH --error=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/test.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`


set -e
export PATH="/nfs/data_chaos/sazzolin/miniconda3/bin:$PATH"
export WANDB_CONFIG_DIR=/home/steve.azzolin/wandb
export WANDB_API_KEY=2cad8a8279143c69ce071f54bf37c1f5a5f4e5ff
export HYDRA_FULL_ERROR=1
eval "$(conda shell.bash hook)"
conda deactivate
conda activate nsd
wandb login


pytest -v .



echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes