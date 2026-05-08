#!/usr/bin/zsh

#SBATCH --job-name=moea_exp
#SBATCH --output=moea_exp_%j.out
#SBATCH --error=moea_exp_%j.err
#SBATCH --time=24:00:00            
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20        
#SBATCH --mem=40G                  
#SBATCH --account=thes2068
#SBATCH --array=0-0

module load Python/3.12.3
source /work/gc570755/myenv/bin/activate

python MOEA_tests.py --run_id ${SLURM_ARRAY_TASK_ID}