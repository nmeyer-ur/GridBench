#!/bin/bash

# copy to GridBench folder
# compile and run all registered templates 

DESCRIPTION="empty"
SUFFIX="armclang"
CXX="armclang++"
BIND=3


# compile and execute

g () {
  echo -e "\n================ $1 ================"
  cd arch/sve/rrii/
  rm -f SVE_rrii.h
  ./intrinsify_rrii.py $1 > SVE_rrii.h
  cd ../../../
  rm -f bench.rrii.sve.intrinsics.${SUFFIX}
  make CXX=${CXX} bench.rrii.sve.intrinsics.${SUFFIX}
  echo -e "------- $1 -------"
  for i in `seq 1 12` ; do OMP_NUM_THREADS=$i numactl --cpunodebind=${BIND} --membind=${BIND} ./bench.rrii.sve.intrinsics.${SUFFIX} 32 100 2> /dev/null | grep XX1 ; done
  echo -e "------- $1 -------"
  echo -e ""
  mv bench.rrii.sve.intrinsics.${SUFFIX} bench.rrii.sve.intrinsics.${SUFFIX}.`basename $1 .h`
}

# main

echo -e "=== description ==="
echo -e "${DESCRIPTION}"

echo -e "\n=== commit ==="
git log | head -n 1

echo -e "\n=== timestamp ==="
date

echo -e "\n=== compiler ==="
${CXX} -v

echo -e "\n=== clock frequency in GHz ==="
cat freq.txt

echo -e "\n=== node ==="
hostname

echo -e "\n=== current working directory ==="
pwd

echo -e "\n=== this script name ==="
echo -e "$0"

echo -e "\n=== this script ==="
cat $0

echo -e "=== end this script ==="

echo -e "\n=== runs ==="

g "SVETemplate7.h"
g "SVETemplate7_constantU_1.h"
g "SVETemplate7_alternativeCMult.h"

