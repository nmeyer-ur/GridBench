
SIMPLEDATA := arch/sse/static_data.cc
USE_LIKWID := false
CXXFLAGS_SVE_O1 := -O1
CXXFLAGS_SVE_O3 := -O3
CXXFLAGS_SVE_NOSCHED_GCC := -O3 -fno-schedule-insns -fno-schedule-insns2

#OMP:=-std=c++11 -DSVM
ifeq (${USE_LIKWID},true)
	OMP:=-std=c++11 -DSVM -DOMP -fopenmp  -I${LIKWID_INCDIR} -DLIKWID_PERFMON 
else
	OMP:=-std=c++11 -DSVM -DOMP -fopenmp
endif

#OMP:=-std=c++11  -O3 -fno-schedule-insns -fno-schedule-insns2 -fno-sched-interblock -DSVM
#OMP:=-std=c++11 -O3

#CXX       := mpicxx-openmpi-devel-clang40
#CXX       := mpiicpc
#CXX       := g++-7
#CXX        := dpcpp
CXX       := g++
#CXX       := mpicxx
#CXX       := clang++-mp-6.0
#CXXCL     := clang++-mp-7.0
#CXXCL     := ${HOME}/QCD/build/bin/clang++
CXXCL     := dpcpp
CXXFLAGSCL:=
LDFLAGSCL:=
CXXFLAGS  := $(OMP)

AVX512_DATA   := arch/avx512/static_data.cc
AVX2_DATA     := arch/avx/static_data_gauge.cc arch/avx/static_data_fermion.cc
AVX_DATA      := arch/avx/static_data_gauge.cc arch/avx/static_data_fermion.cc
SSE_DATA      := arch/sse/static_data.cc
RRII_DATA     := arch/gen64/static_data.cc
RIRI_DATA     := arch/gen64/static_data.cc
RRII_DATA_SVE := arch/sve/rrii/static_data.cc
RIRI_DATA_SVE := arch/sve/riri/static_data.cc


#############################################
# Intel
#############################################
#AVX512_CXXFLAGS  := -DAVX512 -xcore-avx512 $(OMP)
#AVX2_CXXFLAGS    := -DAVX2  -march=core-avx2 -xcore-avx2 $(OMP)
#AVX_CXXFLAGS     := -DAVX1  -mavx -xavx $(OMP)
#SSE_CXXFLAGS     := -DSSE4  -msse4.2 -xsse4.2  $(OMP)
#RRII_CXXFLAGS     := -DRRII  -mavx2 -mfma  $(OMP)

#############################################
# CLANG
#############################################
AVX512_CXXFLAGS  := -DAVX512 -mavx512f -mavx512pf -mavx512er -mavx512cd -O3 $(OMP)
AVX2_CXXFLAGS    := -DAVX2  -mavx2 -mfma $(OMP)
AVX_CXXFLAGS     := -DAVX1  -mavx $(OMP)
SSE_CXXFLAGS     := -DSSE4  -msse4.2  $(OMP)
RRII_CXXFLAGS     := -DRRII  -mavx2 -mfma  $(OMP) -DGEN_SIMD_WIDTH=64
RIRI_CXXFLAGS     := -DRIRI  -mavx2 -mfma  $(OMP) -DGEN_SIMD_WIDTH=64 -g
RRII_CXXFLAGSXX   := -DRRII  -march=knl  $(OMP) -DGEN_SIMD_WIDTH=128


RRII_CXXFLAGS_SVE_GCC               := -DRRII  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DSVE
RRII_CXXFLAGS_SVE_INTRIN_GCC        := -DRRII  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE -g
RIRI_CXXFLAGS_SVE_INTRIN_GCC        := -DRIRI  -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
#RIRI_CXXFLAGS_SVE_INTRIN_GCC        := -march=armv8-a+sve -msve-vector-bits=512  $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
#RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG   := -DRRII  -march=armv8-a+sve $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG   := -DRRII  -mcpu=a64fx $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE
RRII_CXXFLAGS_SVE_INTRIN_FCC        := -DRRII  -Nclang -Kfast $(OMP) -DGEN_SIMD_WIDTH=64 -DINTRIN -DSVE

#############################################
# G++
#############################################
#AVX512_CXXFLAGS  := -DAVX512 -mavx512f -mavx512pf -mavx512er -mavx512cd -O3 $(OMP)
#AVX2_CXXFLAGS    := -DAVX2  -mavx2 -mfma $(OMP)
#AVX_CXXFLAGS     := -DAVX1  -mavx $(OMP)
#SSE_CXXFLAGS     := -DSSE4  -msse4.2  $(OMP)

#Generic options
GENERIC_CXXFLAGS  := -DGEN -O3 -DGEN_SIMD_WIDTH=16 $(OMP)
GENERIC_DATA      := arch/sse/static_data.cc

################################################################################
# NVCC and gpu ; 512 bit vector coalescing
################################################################################

# VOLTA
GPUARCH    := --relocatable-device-code=true -gencode arch=compute_70,code=sm_70

# PASCAL
#GPUARCH    := --relocatable-device-code=true -gencode arch=compute_60,code=sm_60

GPUCC      := nvcc
GPULINK    := nvcc $(GPUARCH)
GPU_CXXFLAGS  := -x cu -DVGPU -DGEN_SIMD_WIDTH=64 -I. -O3 -ccbin g++ -std=c++11 --expt-relaxed-constexpr --expt-extended-lambda $(GPUARCH)  -Xcompiler -fno-strict-aliasing
GPU_LDFLAGS  := -link -ccbin g++
GPU_DATA      := arch/avx512/static_data.cc
################################################################################
LDLIBS    := -lm
ifeq (${USE_LIKWID},true)
	LDFLAGS   := -llikwid -L${LIKWID_LIBDIR}
else
	LDFLAGS   :=
endif

all: bench.avx512 bench.avx2 bench.avx bench.sse bench.gen bench.simple bench.sycl \
	bench.rrii.omp.cpu bench.rrii.sycl.cpu bench.rrii.sycl.cpu.simt  bench.rrii.sycl.gpu bench.rrii.sycl.gpu.simt bench.riri.sycl.gpu.simt

bench.avx512: bench.cc $(AVX512_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(AVX512_CXXFLAGS) bench.cc $(AVX512_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx512

bench.avx2: bench.cc $(AVX2_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(AVX2_CXXFLAGS) bench.cc $(AVX2_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx2

bench.rrii.omp.cpu: bench.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.omp.cpu



# SVE RRII
bench.rrii.sve.gccvectors.gcc: bench.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_GCC) $(CXXFLAGS_SVE_NOSCHED_GCC) bench.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.gccvectors.gcc

bench.rrii.sve.intrinsics.gcc: bench.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.gcc

bench.rrii.sve.intrinsics.armclang: bench.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_ARMCLANG) $(CXXFLAGS_SVE_O3) bench.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.armclang

bench.rrii.sve.intrinsics.fcc: bench.cc $(RRII_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/rrii/SVE_rrii.h
	$(CXX) $(RRII_CXXFLAGS_SVE_INTRIN_FCC) bench.cc $(RRII_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sve.intrinsics.fcc

# SVE RIRI
bench.riri.sve.intrinsics.gcc: bench.cc $(RIRI_DATA_SVE)  WilsonKernelsHand.h Makefile arch/sve/riri/wi.h arch/sve/riri/SVE_riri.h
	$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) $(CXXFLAGS_SVE_O1) bench.cc $(RIRI_DATA_SVE) $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.gcc

#bench.riri.sve.intrinsics.gcc: bench.cc $(RIRI_DATA)  WilsonKernelsHand.h Makefile arch/sve/riri/wi.h arch/sve/riri/SVE_riri.h
#	$(CXX) $(RIRI_CXXFLAGS_SVE_INTRIN_GCC) -O0 bench.cc $(RIRI_DATA) $(LDLIBS) $(LDFLAGS) -o bench.riri.sve.intrinsics.gcc



bench.rrii.sycl.cpu: bench.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sycl.cpu -DGRID_SYCL

bench.rrii.sycl.cpu.simt: bench.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sycl.cpu.simt -DGRID_SYCL -DGRID_SYCL_SIMT

bench.rrii.sycl.gpu: bench.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sycl.gpu  -DGRID_SYCL -DGRID_SYCL_GPU

bench.rrii.sycl.gpu.simt: bench.cc $(RRII_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RRII_CXXFLAGS) bench.cc $(RRII_DATA) $(LDLIBS) $(LDFLAGS) -o bench.rrii.sycl.gpu.simt -DGRID_SYCL -DGRID_SYCL_SIMT -DGRID_SYCL_GPU

bench.riri.sycl.gpu.simt: bench.cc $(RIRI_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RIRI_CXXFLAGS) bench.cc $(RIRI_DATA) $(LDLIBS) $(LDFLAGS) -o bench.riri.sycl.gpu.simt -DGRID_SYCL -DGRID_SYCL_SIMT -DGRID_SYCL_GPU

bench.riri.sycl.cpu.simt: bench.cc $(RIRI_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(RIRI_CXXFLAGS) bench.cc $(RIRI_DATA) $(LDLIBS) $(LDFLAGS) -o bench.riri.sycl.cpu.simt -DGRID_SYCL -DGRID_SYCL_SIMT

bench.avx: bench.cc $(AVX_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(AVX_CXXFLAGS) bench.cc $(AVX_DATA) $(LDLIBS) $(LDFLAGS) -o bench.avx

bench.sse: bench.cc $(SSE_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(SSE_CXXFLAGS) bench.cc $(SSE_DATA) $(LDLIBS) $(LDFLAGS) -o bench.sse

bench.gen: bench.cc $(GENERIC_DATA)  WilsonKernelsHand.h Makefile
	$(CXX) $(GENERIC_CXXFLAGS) bench.cc $(GENERIC_DATA) $(LDLIBS) $(LDFLAGS) -o bench.gen

#	nvcc -x cu -DVGPU -DGEN_SIMD_WIDTH=64 -I. -O3 -ccbin g++ -std=c++11 -Xcompiler -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda --relocatable-device-code=true -gencode arch=compute_60,code=sm_60  -Xcompiler -fno-strict-aliasing -c bench.cc -o bench.gpu.o
bench.gpu: bench.cc $(GPU_DATA)  WilsonKernelsHand.h Makefile
	$(GPUCC) $(GPU_CXXFLAGS) -c bench.cc -o bench.gpu.o
	$(GPUCC) $(GPU_CXXFLAGS) -c $(GPU_DATA) -o data.gpu.o
	$(GPULINK) $(GPU_LDFLAGS) bench.gpu.o data.gpu.o -o bench.gpu $(LDLIBS) $(LDFLAGS)

bench.simple: bench_simple.cc $(SIMPLEDATA) dslash_simple.h Makefile
	$(CXX) $(CXXFLAGS) bench_simple.cc $(SIMPLEDATA) -I/usr/local/Cellar/boost/1.68.0_1/include -o bench.simple

bench.sycl: bench_sycl.cc $(SIMPLEDATA) dslash_simple.h Makefile
	$(CXXCL) -O3 -std=c++17 $(CXXFLAGSCL) bench_sycl.cc $(SIMPLEDATA) $(LDLIBS) $(LDFLAGS) $(LDFLAGSCL) -o bench.sycl

######################
# Build a test from triSYCL distro to check compiler working
######################
#
#parallel_vector_add:
#	clang++-mp-7.0  -std=c++17 -I/Users/ayamaguc/Grid/triSYCL-master/include parallel_vector_add.cpp  -I/usr/local/Cellar/boost/1.68.0_1/include
#

clean:
	rm -f  bench.avx512 bench.avx2 bench.avx bench.sse bench.gen  bench.simple TableGenerate bench.gpu bench.sycl bench.rrii* bench.riri*
	rm -rf  *.dSYM*
	rm -f  *~
