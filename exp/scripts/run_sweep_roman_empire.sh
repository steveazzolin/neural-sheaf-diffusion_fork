#!/bin/bash
#SBATCH -p chaos
#SBATCH -A shared-sml-staff
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --job-name=nsd
#SBATCH -t 1-00
#SBATCH --output=/nfs/data_chaos/sazzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/sweep_heterofilic.txt
#SBATCH --error=/nfs/data_chaos/sazzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/sweep_heterofilic.txt
#SBATCH --ntasks=1
#SBATCH -N 1
#SBATCH --mail-user=steve.azzolin@unitn.it
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

start=`date +%s`

wandb agent mcstewe/sheaf/wwe7ub1w


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes

wandb sync