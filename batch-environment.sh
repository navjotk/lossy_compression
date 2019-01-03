#!/bin/bash

export PYTHONUNBUFFERED=1
export PATH=$PATH:/home/nkukreja1/intel/bin
export DEVITO_ARCH=intel
export DEVITO_OPENMP=1
export DEVITO_LOGGING=DEBUG
source activate devito
export PYTHONPATH=$PYTHONPATH:/home/nkukreja1/pyzfp
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nkukreja1/compression/zfp-0.5.3/lib:/home/nkukreja1/intel/lib
python /home/nkukreja1/devito/scripts/clear_devito_cache.py
