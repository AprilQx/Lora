#!/bin/bash
#SBATCH --job-name=lora_finetuning        
#SBATCH --output=logs/output_%j.log     
#SBATCH --error=logs/error_%j.log       
#SBATCH -p ampere  
#SBATCH --account=MPHIL-DIS-SL2-GPU
#SBATCH --gres=gpu:1 
#SBATCH --nodes=1                       
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=4                                    
#SBATCH --time=2:00:00
#SBATCH --mail-user=xx823@cam.ac.uk
#SBATCH --mail-type=ALL

set -e  
module purge
module load python/3.11.0-icl

mkdir -p logs

cd Lora
source myvnv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Debug Python environment
echo "Python version: $(python --version)"
echo "Python path: $(which python)"
python -c "import torch; print('Torch Loaded:', torch.__version__)"
python -c "import torch; print('CUDA Available:', torch.cuda.is_available())"


# Run script
python experiments/hyperparameter/hyper.py