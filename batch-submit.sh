#!/bin/bash

#cd compression
export PYTHONUNBUFFERED=1
export KMP_HW_SUBSET=28c,1t
export PATH=$PATH:~/intel/bin
export DEVITO_ARCH=intel
export DEVITO_OPENMP=1
export DEVITO_LOGGING=DEBUG
source activate devito2
export PYTHONPATH=$PYTHONPATH:~/checkpointer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nkukreja1/compression/zfp-0.5.3/lib:~/intel/lib
python ~/devito/scripts/clear_devito_cache.py
#[ -z "$CHUNK_SIZE" ] && echo "Need to set CHUNK SIZE" && exit 1;
#[ -z "$ALGO" ] && echo "Need to set ALGO" && exit 1;
#[ -z "$SHUFFLE" ] && echo "Need to set SHUFFLE" && exit 1;
#python run-experiment.py -c $CHUNK_SIZE -s $SHUFFLE -z $ALGO
