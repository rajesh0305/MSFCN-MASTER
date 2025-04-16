# Create new conda environment named 'msfcn'
conda create -n msfcn python=3.8 -y

# Activate the environment
conda activate msfcn

# Install PyTorch with CUDA support (adjust cuda version as needed)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
conda install -y numpy pandas matplotlib scikit-learn tqdm seaborn scipy pillow h5py

# Install OpenCV
conda install -y opencv -c conda-forge

# Install any remaining packages using pip
pip install tensorflow keras