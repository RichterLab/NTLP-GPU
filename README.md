# LES-GPU [![Build Status](https://travis-ci.org/Omegaice/LES-GPU.svg?branch=master)](https://travis-ci.org/Omegaice/LES-GPU) [![codecov](https://codecov.io/gh/Omegaice/LES-GPU/branch/master/graph/badge.svg)](https://codecov.io/gh/Omegaice/LES-GPU)

## Compilation
To build on the CRC machines you must run the following commands:
```
module load cmake/3.2.2
module load mvapich2/2.1-intel-15.0-mlx
module load intel/15.0

mkdir build
cd build
cmake ..
make
```

## SETUP AND RUNNING
To run, make a directory ("case1" or something) where les.run and params.in will go
(i.e., not out of the same directory as les.F)
Make sure all paths in these directories point to the proper locations

