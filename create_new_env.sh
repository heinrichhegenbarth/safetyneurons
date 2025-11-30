#!/bin/bash
#SBATCH --job-name=create_conda_env
#SBATCH --output=create_conda_env.log
#SBATCH --time=04:00:00      # 4 hours
#SBATCH --cpus-per-task=2
#SBATCH --partition=scavenge

# Load Anaconda module
module load Anaconda3

# Initialize Conda for this shell
eval "$(conda shell.bash hook)"

# Define environment path
ENV_PATH="/home/hheg_stli/condaenvs/myenv"

# Create environment if it doesn't exist
if [ ! -d "$ENV_PATH" ]; then
    conda create -p $ENV_PATH python=3.11 -y
    echo "Conda environment created at $ENV_PATH"
fi

# Activate the environment
conda activate $ENV_PATH

# Install only the perft package via pip
pip install --upgrade perft
echo "Package perft installed successfully"

# Verify the package can be imported
python -c "import perft; print('Perft imported successfully')"

echo "Job done :)"
