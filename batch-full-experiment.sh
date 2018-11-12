#!/bin/bash

source /home/nkukreja1/compression/batch-environment.sh
numactl --cpunodebind=0 --membind=0 python /home/nkukreja1/compression/full_experiment.py --ncp $NCP
