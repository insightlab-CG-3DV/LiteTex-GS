#!/bin/bash
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p GPUA800
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=80720M
#SBATCH --time=70:00:00
#SBATCH --output=/gpfs/share/home/2301112015/dashGaussian/gs-texturing-save6/log/%x_%j.out

echo "Start time: `date`"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NNODES: $SLURM_NNODES"
echo "SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_JOB_PARTITION: $SLURM_JOB_PARTITION"

__conda_setup="$('/gpfs/share/software/anaconda/3-2023.09-0/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh" ]; then
        . "/gpfs/share/software/anaconda/3-2023.09-0/etc/profile.d/conda.sh"
    else
        export PATH="/gpfs/share/software/anaconda/3-2023.09-0/bin:$PATH"
    fi
fi
unset __conda_setup

conda activate droidsplat3-from-longsplat2

module load cuda/11.8

if [ -n "$CUDA_HOME" ]; then
    export CUDA_HOME=$CUDA_HOME
else
    CUDA_PATH=$(which nvcc | sed 's/\/bin\/nvcc//')
    export CUDA_HOME=$CUDA_PATH
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

if conda list libstdcxx-ng | grep -q "libstdcxx-ng"; then
    echo "libstdcxx-ng already present, skip install"
else
    conda install -c conda-forge libstdcxx-ng=12.2.0 -y -q || echo "skip libstdcxx-ng install (offline?)"
fi
CONDA_LIB_PATH=$CONDA_PREFIX/lib
LIBSTDCPP_PATH=$(find $CONDA_LIB_PATH -name "libstdc++.so.6" | head -1)
export LD_PRELOAD=$LIBSTDCPP_PATH:$LD_PRELOAD

echo "========== Environment =========="
echo "Conda environment: $CONDA_DEFAULT_ENV"
echo "CUDA_HOME: $CUDA_HOME"
echo "NVCC version: $(nvcc --version | head -1)"
echo "libstdc++ path: $LIBSTDCPP_PATH"
echo "GLIBCXX_3.4.29 entries: $(strings $LIBSTDCPP_PATH | grep -c "GLIBCXX_3.4.29")"
echo "=============================="

cd /gpfs/share/home/2301112015/dashGaussian/gs-texturing-save6/submodules/diff-gaussian-rasterization-texture
pwd
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

export TORCH_CUDA_ARCH_LIST="8.0"  # A800 (SM80)
python setup.py build_ext --inplace || true

cd ../simple-knn
python setup.py build_ext --inplace || true

cd ../../
export PYTHONPATH="/gpfs/share/home/2301112015/dashGaussian/gs-texturing-save6/submodules/diff-gaussian-rasterization-texture:/gpfs/share/home/2301112015/dashGaussian/gs-texturing-save6/submodules/simple-knn:$PYTHONPATH"

echo "========== Build outputs =========="
ls -lh submodules/diff-gaussian-rasterization-texture/diff_gaussian_rasterization_texture | head -n 50 || true
ls -lh submodules/simple-knn/simple_knn | head -n 50 || true
echo "================================"
