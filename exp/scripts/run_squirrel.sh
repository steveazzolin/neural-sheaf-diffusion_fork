#!/bin/bash
#SBATCH -p long-disi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=nsd
#SBATCH -t 2-00:00
#SBATCH --output=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/squirrel.txt
#SBATCH --error=/home/steve.azzolin/sheafs/neural-sheaf-diffusion_fork/sbatch_outputs/squirrel.txt
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
export ENTITY=mcstewe
eval "$(conda shell.bash hook)"
conda deactivate
conda activate nsd
wandb login


python -m exp.run \
    --add_lp=True \
    --d=3 \
    --dataset=squirrel \
    --dropout=0 \
    --early_stopping=100 \
    --epochs=1000 \
    --folds=10 \
    --hidden_channels=32 \
    --input_dropout=0.7 \
    --layers=5 \
    --lr=0.01 \
    --model=BundleSheaf \
    --orth=householder \
    --second_linear=True \
    --weight_decay=0.00011215791366362148 \
    --left_weights=True \
    --right_weights=True \
    --use_act=True \
    --normalised=True \
    --edge_weights=True \
    --stop_strategy=acc \
    --entity="${ENTITY}" 


echo DONE
end=`date +%s`
runtime=$( echo "$end - $start" | bc -l )
runtime=$((runtime / 60))
echo Execution lasted $runtime minutes