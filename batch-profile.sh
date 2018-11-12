#!/bin/sh

source /home/nkukreja1/compression/batch-environment.sh
source /home/nkukreja1/intel/vtune_amplifier/amplxe-vars.sh
amplxe-cl -collect memory-access -result-dir r004hs -- python /home/nkukreja1/compression/full_experiment.py --ncp $NCP 
