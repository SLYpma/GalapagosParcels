#!/bin/bash
#SBATCH -t 10:00:00
#SBATCH -N 1
#SBATCH -J 4km_stokes


echo 'Executing program ...'

python galapagosrun_bwd_4km_2008_wstokes.py

echo 'Finished computation.'
