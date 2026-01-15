# NGSolve on Device with NGSCuda

### Getting started on Musica

log in to musica
```bash
# ssh-keygen -R musica.vie.asc.ac.at
ssh username@musica.vie.asc.ac.at
```

building script 
```bash

#!/bin/sh
set -e

ml --force purge
ml load ASC/2023.06

# load the build environment module
ml load buildenv/default-foss-2023a

# load additional build deps
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0 SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0

# unload pybind11 
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

WORKING_DIR=$(realpath "${PWD}")
VENV="${WORKING_DIR}/ngs"
SOURCES="${WORKING_DIR}/src/ngsolve"

virtualenv "${VENV}"
source ${VENV}/bin/activate

# clone from TU Wien GitLab (LATEST, default branch)
if [ ! -d "${SOURCES}/.git" ]; then
    mkdir -p "$(dirname "${SOURCES}")"
    git clone --recurse-submodules https://github.com/NGSolve/ngsolve.git "${SOURCES}"
fi

BUILD_DIR="${WORKING_DIR}/build/ngsolve"
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${SOURCES}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DUSE_SUPERBUILD=ON \
  -DUSE_OCC=ON \
  -DUSE_CCACHE=ON \
  -DCMAKE_INSTALL_PREFIX="${WORKING_DIR}/install" \
  -DUSE_CUDA=ON \
  -DUSE_GUI=OFF \
  -DCMAKE_CUDA_ARCHITECTURES="90" \
  -DUSE_UMFPACK=OFF \
  -DBUILD_STUB_FILES=OFF

make -j 8
make install

echo
echo "=== Build done ==="
echo "Install prefix: ${WORKING_DIR}/install"
echo "Venv:           ${VENV}"


```

run with 
```bash
bash build_ngsolve_musica.sh
```

additionally on a login node 
```bash
source ./ngs/bin/activate

python -m pip install -U pip
python -m pip install numpy
```

submit on musica gpu node with 
```bash

#SBATCH --job-name=myjob
#SBATCH --gres=gpu:1
#SBATCH -p zen4_0768_h100x4
#SBATCH --qos=zen4_0768_h100x4
#SBATCH --threads-per-core=1
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err


ml --force purge
ml load ASC/2023.06

# load the build environment module
ml load buildenv/default-foss-2023a

# load additional build deps
ml load CMake/3.26.3-GCCcore-12.3.0 CUDA/12.9.0 SciPy-bundle/2023.07-gfbf-2023a occt/7.8.0-GCCcore-12.3.0
ml unload pybind11/2.11.1-GCCcore-12.3.0 || true

# use the same folder you built in
WORKING_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
VENV="$WORKING_DIR/ngs"
PREFIX="$WORKING_DIR/install"

source "$VENV/bin/activate"

export PYTHONNOUSERSITE=1
unset PYTHONPATH || true

# make CUDA runtime libs visible (fixes libcusparse.so.12)
if [ -n "${EBROOTCUDA:-}" ] && [ -d "$EBROOTCUDA/lib64" ]; then
  export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
fi

PYDIR=$(find "$PREFIX" -maxdepth 6 -type d \( -name site-packages -o -name dist-packages \) | head -n 1)

export PYTHONPATH="$PYDIR:${PYTHONPATH:-}"
export PATH="$PREFIX/bin:$PATH"

echo "WORKING_DIR=$WORKING_DIR"
echo "VENV=$VENV"
echo "PREFIX=$PREFIX"
echo "PYDIR=$PYDIR"
echo

which python
python -V
nvidia-smi
echo

echo "Job started at $(date)"
echo "Running on host $(hostname)"
echo

python - <<'PY'
import ngsolve
import ngsolve.ngscuda as ngscuda
print("NGSolve version:", getattr(ngsolve, "__version__", "unknown"))
print("ngsolve file:", ngsolve.__file__)
print("ngscuda file:", ngscuda.__file__)
print("Has CudaGraph:", hasattr(ngscuda, "CudaGraph"))
PY

echo
echo "Job finished at $(date)"
```

submit the job on a compute node with 
```bash
sbatch submit.sh
```

look for outcome for example with nano in generated 
 - ngscuda_graph-<jobid>.out
 - ngscuda_graph-<jobid>.err





