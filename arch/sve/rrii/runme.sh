#!/bin/bash

# copy to GridBench folder
# compile and run all registered templates 

SUFFIX="gcc"
CXX="g++"
BIND=3

# compile and execute

g () {
  echo "================ $1 ================"
  cd arch/sve/rrii/
  rm -f SVE_rrii.h
  ./intrinsify_rrii.py $1 > SVE_rrii.h
  cd ../../../
  rm -f bench.rrii.sve.intrinsics.${SUFFIX}
  make CXX=${CXX} bench.rrii.sve.intrinsics.${SUFFIX}
  echo "------- $1 -------"
  for i in `seq 1 12` ; do OMP_NUM_THREADS=$i numactl --cpunodebind=${BIND} --membind=${BIND} ./bench.rrii.sve.intrinsics.${SUFFIX} 32 100 2> /dev/null | grep XX1 ; done
  echo "------- $1 -------"
  echo ""
  mv bench.rrii.sve.intrinsics.${SUFFIX} bench.rrii.sve.intrinsics.${SUFFIX}.`basename $1 .h`
}

# main

g "SVETemplate0.h"
g "SVETemplate1.h"
g "SVETemplate2.h"
g "SVETemplate3.h"
g "SVETemplate3_constantU_1.h"
g "SVETemplate3_constantU_2.h"
g "SVETemplate3_constantU_3.h"
g "SVETemplate3_noLoadUInInnerLoop.h"
g "SVETemplate4.h"
g "SVETemplate5_integratedPF.h"
g "SVETemplate5.h"

