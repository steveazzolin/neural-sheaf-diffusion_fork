#!/bin/bash
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --job-name=nsd
#SBATCH -t 0-00:15
#SBATCH --output=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/sweep_heterofilic2.txt
#SBATCH --error=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/sweep_heterofilic2.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`

wandb agent mcstewe/sheaf/bmof5ulj


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes

wandb sync