// intrinsify
// arch                A64FX / 512-bit SVE
// vector length       64 bytes, 512 bits
// header file         arm_sve.h
// float type          float64_t
// float* typecast     (float64_t*)
// simd type           svfloat64_t

#include <arm_sve.h>
    
#include <stdio.h>
#include <arm_sve.h>

#pragma once

#ifdef GRID_SYCL
#include <CL/sycl.hpp>
#include <CL/sycl/usm.hpp>
#endif

#ifdef GRID_SYCL_SIMT
//#define synchronise(A) item.barrier(cl::sycl::access::fence_space::local_space);
#define synchronise(A)
#else
#define synchronise(A)
#endif

#if defined(GRID_SYCL_SIMT) || defined(GRID_NVCC)
#define LOAD_CHIMU(ptype)		\
  { const SiteSpinor & ref (in[offset]);	\
Chimu_00=coalescedReadPermute<ptype>(ref[0][0],perm,mylane);	\
Chimu_01=coalescedReadPermute<ptype>(ref[0][1],perm,mylane);	\
Chimu_02=coalescedReadPermute<ptype>(ref[0][2],perm,mylane);	\
Chimu_10=coalescedReadPermute<ptype>(ref[1][0],perm,mylane);	\
Chimu_11=coalescedReadPermute<ptype>(ref[1][1],perm,mylane);	\
Chimu_12=coalescedReadPermute<ptype>(ref[1][2],perm,mylane);	\
Chimu_20=coalescedReadPermute<ptype>(ref[2][0],perm,mylane);	\
Chimu_21=coalescedReadPermute<ptype>(ref[2][1],perm,mylane);	\
Chimu_22=coalescedReadPermute<ptype>(ref[2][2],perm,mylane);	\
Chimu_30=coalescedReadPermute<ptype>(ref[3][0],perm,mylane);	\
Chimu_31=coalescedReadPermute<ptype>(ref[3][1],perm,mylane);	\
Chimu_32=coalescedReadPermute<ptype>(ref[3][2],perm,mylane);}

#define PERMUTE_DIR(dir) ;

#else
#define LOAD_CHIMU(ptype)		\
  { const SiteSpinor & ref (in[offset]);	base = (uint64_t)ref; \
    Chimu_00_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)));\
    Chimu_00_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)));\
    Chimu_01_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)));\
    Chimu_01_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)));\
    Chimu_02_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)));\
    Chimu_02_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)));\
    Chimu_10_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)));\
    Chimu_10_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)));\
    Chimu_11_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1)));\
    Chimu_11_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1 + 1)));\
    Chimu_12_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2)));\
    Chimu_12_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2 + 1)));\
    Chimu_20_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0)));\
    Chimu_20_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0 + 1)));\
    Chimu_21_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1)));\
    Chimu_21_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1 + 1)));\
    Chimu_22_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2)));\
    Chimu_22_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2 + 1)));\
    Chimu_30_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 0)));\
    Chimu_30_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 0 + 1)));\
    Chimu_31_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 1)));\
    Chimu_31_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 1 + 1)));\
    Chimu_32_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 2)));\
    Chimu_32_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 2 + 1)));}

/*
#define PERMUTE_DIR(dir)			\
      permute##dir(Chi_00,Chi_00);\
      permute##dir(Chi_01,Chi_01);\
      permute##dir(Chi_02,Chi_02);\
      permute##dir(Chi_10,Chi_10);\
      permute##dir(Chi_11,Chi_11);\
      permute##dir(Chi_12,Chi_12);
*/

#define PERMUTE_DIR(dir)

#endif


#define MULT_2SPIN(A)\
  { auto & ref(U[sU][A]); base = (uint64_t)ref;	\
    U_00_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)));\
    U_00_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)));\
    U_10_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)));\
    U_10_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)));\
    U_20_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0)));\
    U_20_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0 + 1)));\
    U_01_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)));\
    U_01_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)));\
    U_11_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1)));\
    U_11_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1 + 1)));\
    U_21_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1)));\
    U_21_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1 + 1)));\
    UChi_00_re = svmul_x(pg1, U_00_re, Chi_00_re);\
    UChi_00_im = svmul_x(pg1, U_00_re, Chi_00_im);\
    UChi_00_re = svmls_x(pg1, UChi_00_re, U_00_im, Chi_00_im);\
    UChi_00_im = svmla_x(pg1, UChi_00_im, U_00_im, Chi_00_re);\
    UChi_10_re = svmul_x(pg1, U_00_re, Chi_10_re);\
    UChi_10_im = svmul_x(pg1, U_00_re, Chi_10_im);\
    UChi_10_re = svmls_x(pg1, UChi_10_re, U_00_im, Chi_10_im);\
    UChi_10_im = svmla_x(pg1, UChi_10_im, U_00_im, Chi_10_re);\
    UChi_01_re = svmul_x(pg1, U_10_re, Chi_00_re);\
    UChi_01_im = svmul_x(pg1, U_10_re, Chi_00_im);\
    UChi_01_re = svmls_x(pg1, UChi_01_re, U_10_im, Chi_00_im);\
    UChi_01_im = svmla_x(pg1, UChi_01_im, U_10_im, Chi_00_re);\
    UChi_11_re = svmul_x(pg1, U_10_re, Chi_10_re);\
    UChi_11_im = svmul_x(pg1, U_10_re, Chi_10_im);\
    UChi_11_re = svmls_x(pg1, UChi_11_re, U_10_im, Chi_10_im);\
    UChi_11_im = svmla_x(pg1, UChi_11_im, U_10_im, Chi_10_re);\
    UChi_02_re = svmul_x(pg1, U_20_re, Chi_00_re);\
    UChi_02_im = svmul_x(pg1, U_20_re, Chi_00_im);\
    UChi_02_re = svmls_x(pg1, UChi_02_re, U_20_im, Chi_00_im);\
    UChi_02_im = svmla_x(pg1, UChi_02_im, U_20_im, Chi_00_re);\
    UChi_12_re = svmul_x(pg1, U_20_re, Chi_10_re);\
    UChi_12_im = svmul_x(pg1, U_20_re, Chi_10_im);\
    UChi_12_re = svmls_x(pg1, UChi_12_re, U_20_im, Chi_10_im);\
    UChi_12_im = svmla_x(pg1, UChi_12_im, U_20_im, Chi_10_re);\
    UChi_00_re = svmla_x(pg1, UChi_00_re, U_01_re, Chi_01_re);\
    UChi_00_im = svmla_x(pg1, UChi_00_im, U_01_re, Chi_01_im);\
    UChi_00_re = svmls_x(pg1, UChi_00_re, U_01_im, Chi_01_im);\
    UChi_00_im = svmla_x(pg1, UChi_00_im, U_01_im, Chi_01_re);\
    UChi_10_re = svmla_x(pg1, UChi_10_re, U_01_re, Chi_11_re);\
    UChi_10_im = svmla_x(pg1, UChi_10_im, U_01_re, Chi_11_im);\
    UChi_10_re = svmls_x(pg1, UChi_10_re, U_01_im, Chi_11_im);\
    UChi_10_im = svmla_x(pg1, UChi_10_im, U_01_im, Chi_11_re);\
    UChi_01_re = svmla_x(pg1, UChi_01_re, U_11_re, Chi_01_re);\
    UChi_01_im = svmla_x(pg1, UChi_01_im, U_11_re, Chi_01_im);\
    UChi_01_re = svmls_x(pg1, UChi_01_re, U_11_im, Chi_01_im);\
    UChi_01_im = svmla_x(pg1, UChi_01_im, U_11_im, Chi_01_re);\
    UChi_11_re = svmla_x(pg1, UChi_11_re, U_11_re, Chi_11_re);\
    UChi_11_im = svmla_x(pg1, UChi_11_im, U_11_re, Chi_11_im);\
    UChi_11_re = svmls_x(pg1, UChi_11_re, U_11_im, Chi_11_im);\
    UChi_11_im = svmla_x(pg1, UChi_11_im, U_11_im, Chi_11_re);\
    UChi_02_re = svmla_x(pg1, UChi_02_re, U_21_re, Chi_01_re);\
    UChi_02_im = svmla_x(pg1, UChi_02_im, U_21_re, Chi_01_im);\
    UChi_02_re = svmls_x(pg1, UChi_02_re, U_21_im, Chi_01_im);\
    UChi_02_im = svmla_x(pg1, UChi_02_im, U_21_im, Chi_01_re);\
    UChi_12_re = svmla_x(pg1, UChi_12_re, U_21_re, Chi_11_re);\
    UChi_12_im = svmla_x(pg1, UChi_12_im, U_21_re, Chi_11_im);\
    UChi_12_re = svmls_x(pg1, UChi_12_re, U_21_im, Chi_11_im);\
    UChi_12_im = svmla_x(pg1, UChi_12_im, U_21_im, Chi_11_re);\
    U_00_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)));\
    U_00_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)));\
    U_10_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2)));\
    U_10_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2 + 1)));\
    U_20_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2)));\
    U_20_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2 + 1)));\
    UChi_00_re = svmla_x(pg1, UChi_00_re, U_00_re, Chi_02_re);\
    UChi_00_im = svmla_x(pg1, UChi_00_im, U_00_re, Chi_02_im);\
    UChi_00_re = svmls_x(pg1, UChi_00_re, U_00_im, Chi_02_im);\
    UChi_00_im = svmla_x(pg1, UChi_00_im, U_00_im, Chi_02_re);\
    UChi_10_re = svmla_x(pg1, UChi_10_re, U_00_re, Chi_12_re);\
    UChi_10_im = svmla_x(pg1, UChi_10_im, U_00_re, Chi_12_im);\
    UChi_10_re = svmls_x(pg1, UChi_10_re, U_00_im, Chi_12_im);\
    UChi_10_im = svmla_x(pg1, UChi_10_im, U_00_im, Chi_12_re);\
    UChi_01_re = svmla_x(pg1, UChi_01_re, U_10_re, Chi_02_re);\
    UChi_01_im = svmla_x(pg1, UChi_01_im, U_10_re, Chi_02_im);\
    UChi_01_re = svmls_x(pg1, UChi_01_re, U_10_im, Chi_02_im);\
    UChi_01_im = svmla_x(pg1, UChi_01_im, U_10_im, Chi_02_re);\
    UChi_11_re = svmla_x(pg1, UChi_11_re, U_10_re, Chi_12_re);\
    UChi_11_im = svmla_x(pg1, UChi_11_im, U_10_re, Chi_12_im);\
    UChi_11_re = svmls_x(pg1, UChi_11_re, U_10_im, Chi_12_im);\
    UChi_11_im = svmla_x(pg1, UChi_11_im, U_10_im, Chi_12_re);\
    UChi_02_re = svmla_x(pg1, UChi_02_re, U_20_re, Chi_02_re);\
    UChi_02_im = svmla_x(pg1, UChi_02_im, U_20_re, Chi_02_im);\
    UChi_02_re = svmls_x(pg1, UChi_02_re, U_20_im, Chi_02_im);\
    UChi_02_im = svmla_x(pg1, UChi_02_im, U_20_im, Chi_02_re);\
    UChi_12_re = svmla_x(pg1, UChi_12_re, U_20_re, Chi_12_re);\
    UChi_12_im = svmla_x(pg1, UChi_12_im, U_20_re, Chi_12_im);\
    UChi_12_re = svmls_x(pg1, UChi_12_re, U_20_im, Chi_12_im);\
    UChi_12_im = svmla_x(pg1, UChi_12_im, U_20_im, Chi_12_re);}

//      hspin(0)=fspin(0)+timesI(fspin(3));
//      hspin(1)=fspin(1)+timesI(fspin(2));
#define XP_PROJ \
    Chi_00_re = svsub_x(pg1, Chimu_00_re, Chimu_30_im);\
    Chi_00_im = svadd_x(pg1, Chimu_00_im, Chimu_30_re);\
    Chi_01_re = svsub_x(pg1, Chimu_01_re, Chimu_31_im);\
    Chi_01_im = svadd_x(pg1, Chimu_01_im, Chimu_31_re);\
    Chi_02_re = svsub_x(pg1, Chimu_02_re, Chimu_32_im);\
    Chi_02_im = svadd_x(pg1, Chimu_02_im, Chimu_32_re);\
    Chi_10_re = svsub_x(pg1, Chimu_10_re, Chimu_20_im);\
    Chi_10_im = svadd_x(pg1, Chimu_10_im, Chimu_20_re);\
    Chi_11_re = svsub_x(pg1, Chimu_11_re, Chimu_21_im);\
    Chi_11_im = svadd_x(pg1, Chimu_11_im, Chimu_21_re);\
    Chi_12_re = svsub_x(pg1, Chimu_12_re, Chimu_22_im);\
    Chi_12_im = svadd_x(pg1, Chimu_12_im, Chimu_22_re);

#define YP_PROJ \
    Chi_00_re = svsub_x(pg1, Chimu_00_re, Chimu_30_re);\
    Chi_00_im = svsub_x(pg1, Chimu_00_im, Chimu_30_im);\
    Chi_01_re = svsub_x(pg1, Chimu_01_re, Chimu_31_re);\
    Chi_01_im = svsub_x(pg1, Chimu_01_im, Chimu_31_im);\
    Chi_02_re = svsub_x(pg1, Chimu_02_re, Chimu_32_re);\
    Chi_02_im = svsub_x(pg1, Chimu_02_im, Chimu_32_im);\
    Chi_10_re = svadd_x(pg1, Chimu_10_re, Chimu_20_re);\
    Chi_10_im = svadd_x(pg1, Chimu_10_im, Chimu_20_im);\
    Chi_11_re = svadd_x(pg1, Chimu_11_re, Chimu_21_re);\
    Chi_11_im = svadd_x(pg1, Chimu_11_im, Chimu_21_im);\
    Chi_12_re = svadd_x(pg1, Chimu_12_re, Chimu_22_re);\
    Chi_12_im = svadd_x(pg1, Chimu_12_im, Chimu_22_im);

#define ZP_PROJ \
    Chi_00_re = svsub_x(pg1, Chimu_00_re, Chimu_20_im);\
    Chi_00_im = svadd_x(pg1, Chimu_00_im, Chimu_20_re);\
    Chi_01_re = svsub_x(pg1, Chimu_01_re, Chimu_21_im);\
    Chi_01_im = svadd_x(pg1, Chimu_01_im, Chimu_21_re);\
    Chi_02_re = svsub_x(pg1, Chimu_02_re, Chimu_22_im);\
    Chi_02_im = svadd_x(pg1, Chimu_02_im, Chimu_22_re);\
    Chi_10_re = svadd_x(pg1, Chimu_10_re, Chimu_30_im);\
    Chi_10_im = svsub_x(pg1, Chimu_10_im, Chimu_30_re);\
    Chi_11_re = svadd_x(pg1, Chimu_11_re, Chimu_31_im);\
    Chi_11_im = svsub_x(pg1, Chimu_11_im, Chimu_31_re);\
    Chi_12_re = svadd_x(pg1, Chimu_12_re, Chimu_32_im);\
    Chi_12_im = svsub_x(pg1, Chimu_12_im, Chimu_32_re);

#define TP_PROJ \
    Chi_00_re = svadd_x(pg1, Chimu_00_re, Chimu_20_re);\
    Chi_00_im = svadd_x(pg1, Chimu_00_im, Chimu_20_im);\
    Chi_01_re = svadd_x(pg1, Chimu_01_re, Chimu_21_re);\
    Chi_01_im = svadd_x(pg1, Chimu_01_im, Chimu_21_im);\
    Chi_02_re = svadd_x(pg1, Chimu_02_re, Chimu_22_re);\
    Chi_02_im = svadd_x(pg1, Chimu_02_im, Chimu_22_im);\
    Chi_10_re = svadd_x(pg1, Chimu_10_re, Chimu_30_re);\
    Chi_10_im = svadd_x(pg1, Chimu_10_im, Chimu_30_im);\
    Chi_11_re = svadd_x(pg1, Chimu_11_re, Chimu_31_re);\
    Chi_11_im = svadd_x(pg1, Chimu_11_im, Chimu_31_im);\
    Chi_12_re = svadd_x(pg1, Chimu_12_re, Chimu_32_re);\
    Chi_12_im = svadd_x(pg1, Chimu_12_im, Chimu_32_im);


//      hspin(0)=fspin(0)-timesI(fspin(3));
//      hspin(1)=fspin(1)-timesI(fspin(2));
#define XM_PROJ \
    Chi_00_re = svadd_x(pg1, Chimu_00_re, Chimu_30_im);\
    Chi_00_im = svsub_x(pg1, Chimu_00_im, Chimu_30_re);\
    Chi_01_re = svadd_x(pg1, Chimu_01_re, Chimu_31_im);\
    Chi_01_im = svsub_x(pg1, Chimu_01_im, Chimu_31_re);\
    Chi_02_re = svadd_x(pg1, Chimu_02_re, Chimu_32_im);\
    Chi_02_im = svsub_x(pg1, Chimu_02_im, Chimu_32_re);\
    Chi_10_re = svadd_x(pg1, Chimu_10_re, Chimu_20_im);\
    Chi_10_im = svsub_x(pg1, Chimu_10_im, Chimu_20_re);\
    Chi_11_re = svadd_x(pg1, Chimu_11_re, Chimu_21_im);\
    Chi_11_im = svsub_x(pg1, Chimu_11_im, Chimu_21_re);\
    Chi_12_re = svadd_x(pg1, Chimu_12_re, Chimu_22_im);\
    Chi_12_im = svsub_x(pg1, Chimu_12_im, Chimu_22_re);

#define YM_PROJ \
    Chi_00_re = svadd_x(pg1, Chimu_00_re, Chimu_30_re);\
    Chi_00_im = svadd_x(pg1, Chimu_00_im, Chimu_30_im);\
    Chi_01_re = svadd_x(pg1, Chimu_01_re, Chimu_31_re);\
    Chi_01_im = svadd_x(pg1, Chimu_01_im, Chimu_31_im);\
    Chi_02_re = svadd_x(pg1, Chimu_02_re, Chimu_32_re);\
    Chi_02_im = svadd_x(pg1, Chimu_02_im, Chimu_32_im);\
    Chi_10_re = svsub_x(pg1, Chimu_10_re, Chimu_20_re);\
    Chi_10_im = svsub_x(pg1, Chimu_10_im, Chimu_20_im);\
    Chi_11_re = svsub_x(pg1, Chimu_11_re, Chimu_21_re);\
    Chi_11_im = svsub_x(pg1, Chimu_11_im, Chimu_21_im);\
    Chi_12_re = svsub_x(pg1, Chimu_12_re, Chimu_22_re);\
    Chi_12_im = svsub_x(pg1, Chimu_12_im, Chimu_22_im);

#define ZM_PROJ \
    Chi_00_re = svadd_x(pg1, Chimu_00_re, Chimu_20_im);\
    Chi_00_im = svsub_x(pg1, Chimu_00_im, Chimu_20_re);\
    Chi_01_re = svadd_x(pg1, Chimu_01_re, Chimu_21_im);\
    Chi_01_im = svsub_x(pg1, Chimu_01_im, Chimu_21_re);\
    Chi_02_re = svadd_x(pg1, Chimu_02_re, Chimu_22_im);\
    Chi_02_im = svsub_x(pg1, Chimu_02_im, Chimu_22_re);\
    Chi_10_re = svsub_x(pg1, Chimu_10_re, Chimu_30_im);\
    Chi_10_im = svadd_x(pg1, Chimu_10_im, Chimu_30_re);\
    Chi_11_re = svsub_x(pg1, Chimu_11_re, Chimu_31_im);\
    Chi_11_im = svadd_x(pg1, Chimu_11_im, Chimu_31_re);\
    Chi_12_re = svsub_x(pg1, Chimu_12_re, Chimu_32_im);\
    Chi_12_im = svadd_x(pg1, Chimu_12_im, Chimu_32_re);

#define TM_PROJ \
    Chi_00_re = svsub_x(pg1, Chimu_00_re, Chimu_20_re);\
    Chi_00_im = svsub_x(pg1, Chimu_00_im, Chimu_20_im);\
    Chi_01_re = svsub_x(pg1, Chimu_01_re, Chimu_21_re);\
    Chi_01_im = svsub_x(pg1, Chimu_01_im, Chimu_21_im);\
    Chi_02_re = svsub_x(pg1, Chimu_02_re, Chimu_22_re);\
    Chi_02_im = svsub_x(pg1, Chimu_02_im, Chimu_22_im);\
    Chi_10_re = svsub_x(pg1, Chimu_10_re, Chimu_30_re);\
    Chi_10_im = svsub_x(pg1, Chimu_10_im, Chimu_30_im);\
    Chi_11_re = svsub_x(pg1, Chimu_11_re, Chimu_31_re);\
    Chi_11_im = svsub_x(pg1, Chimu_11_im, Chimu_31_im);\
    Chi_12_re = svsub_x(pg1, Chimu_12_re, Chimu_32_re);\
    Chi_12_im = svsub_x(pg1, Chimu_12_im, Chimu_32_im);

//      fspin(0)=hspin(0);
//      fspin(1)=hspin(1);
//      fspin(2)=timesMinusI(hspin(1));
//      fspin(3)=timesMinusI(hspin(0));

#define XP_RECON\
    result_00_re = UChi_00_re;\
    result_00_im = UChi_00_im;\
    result_01_re = UChi_01_re;\
    result_01_im = UChi_01_im;\
    result_02_re = UChi_02_re;\
    result_02_im = UChi_02_im;\
    result_10_re = UChi_10_re;\
    result_10_im = UChi_10_im;\
    result_11_re = UChi_11_re;\
    result_11_im = UChi_11_im;\
    result_12_re = UChi_12_re;\
    result_12_im = UChi_12_im;\
    result_20_re = UChi_10_im;\
    result_20_im = svneg_x(pg1, UChi_10_re);\
    result_21_re = UChi_11_im;\
    result_21_im = svneg_x(pg1, UChi_11_re);\
    result_22_re = UChi_12_im;\
    result_22_im = svneg_x(pg1, UChi_12_re);\
    result_30_re = UChi_00_im;\
    result_30_im = svneg_x(pg1, UChi_00_re);\
    result_31_re = UChi_01_im;\
    result_31_im = svneg_x(pg1, UChi_01_re);\
    result_32_re = UChi_02_im;\
    result_32_im = svneg_x(pg1, UChi_02_re);

#define XP_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svadd_x(pg1, result_20_re, UChi_10_im);\
    result_20_im = svsub_x(pg1, result_20_im, UChi_10_re);\
    result_21_re = svadd_x(pg1, result_21_re, UChi_11_im);\
    result_21_im = svsub_x(pg1, result_21_im, UChi_11_re);\
    result_22_re = svadd_x(pg1, result_22_re, UChi_12_im);\
    result_22_im = svsub_x(pg1, result_22_im, UChi_12_re);\
    result_30_re = svadd_x(pg1, result_30_re, UChi_00_im);\
    result_30_im = svsub_x(pg1, result_30_im, UChi_00_re);\
    result_31_re = svadd_x(pg1, result_31_re, UChi_01_im);\
    result_31_im = svsub_x(pg1, result_31_im, UChi_01_re);\
    result_32_re = svadd_x(pg1, result_32_re, UChi_02_im);\
    result_32_im = svsub_x(pg1, result_32_im, UChi_02_re);

#define XM_RECON\
    result_00_re = UChi_00_re;\
    result_00_im = UChi_00_im;\
    result_01_re = UChi_01_re;\
    result_01_im = UChi_01_im;\
    result_02_re = UChi_02_re;\
    result_02_im = UChi_02_im;\
    result_10_re = UChi_10_re;\
    result_10_im = UChi_10_im;\
    result_11_re = UChi_11_re;\
    result_11_im = UChi_11_im;\
    result_12_re = UChi_12_re;\
    result_12_im = UChi_12_im;\
    result_20_re = svneg_x(pg1, UChi_10_im);\
    result_20_im = UChi_10_re;\
    result_21_re = svneg_x(pg1, UChi_11_im);\
    result_21_im = UChi_11_re;\
    result_22_re = svneg_x(pg1, UChi_12_im);\
    result_22_im = UChi_12_re;\
    result_30_re = svneg_x(pg1, UChi_00_im);\
    result_30_im = UChi_00_re;\
    result_31_re = svneg_x(pg1, UChi_01_im);\
    result_31_im = UChi_01_re;\
    result_32_re = svneg_x(pg1, UChi_02_im);\
    result_32_im = UChi_02_re;

#define XM_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svsub_x(pg1, result_20_re, UChi_10_im);\
    result_20_im = svadd_x(pg1, result_20_im, UChi_10_re);\
    result_21_re = svsub_x(pg1, result_21_re, UChi_11_im);\
    result_21_im = svadd_x(pg1, result_21_im, UChi_11_re);\
    result_22_re = svsub_x(pg1, result_22_re, UChi_12_im);\
    result_22_im = svadd_x(pg1, result_22_im, UChi_12_re);\
    result_30_re = svsub_x(pg1, result_30_re, UChi_00_im);\
    result_30_im = svadd_x(pg1, result_30_im, UChi_00_re);\
    result_31_re = svsub_x(pg1, result_31_re, UChi_01_im);\
    result_31_im = svadd_x(pg1, result_31_im, UChi_01_re);\
    result_32_re = svsub_x(pg1, result_32_re, UChi_02_im);\
    result_32_im = svadd_x(pg1, result_32_im, UChi_02_re);

#define YP_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svadd_x(pg1, result_20_re, UChi_10_re);\
    result_20_im = svadd_x(pg1, result_20_im, UChi_10_im);\
    result_21_re = svadd_x(pg1, result_21_re, UChi_11_re);\
    result_21_im = svadd_x(pg1, result_21_im, UChi_11_im);\
    result_22_re = svadd_x(pg1, result_22_re, UChi_12_re);\
    result_22_im = svadd_x(pg1, result_22_im, UChi_12_im);\
    result_30_re = svsub_x(pg1, result_30_re, UChi_00_re);\
    result_30_im = svsub_x(pg1, result_30_im, UChi_00_im);\
    result_31_re = svsub_x(pg1, result_31_re, UChi_01_re);\
    result_31_im = svsub_x(pg1, result_31_im, UChi_01_im);\
    result_32_re = svsub_x(pg1, result_32_re, UChi_02_re);\
    result_32_im = svsub_x(pg1, result_32_im, UChi_02_im);

#define YM_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svsub_x(pg1, result_20_re, UChi_10_re);\
    result_20_im = svsub_x(pg1, result_20_im, UChi_10_im);\
    result_21_re = svsub_x(pg1, result_21_re, UChi_11_re);\
    result_21_im = svsub_x(pg1, result_21_im, UChi_11_im);\
    result_22_re = svsub_x(pg1, result_22_re, UChi_12_re);\
    result_22_im = svsub_x(pg1, result_22_im, UChi_12_im);\
    result_30_re = svadd_x(pg1, result_30_re, UChi_00_re);\
    result_30_im = svadd_x(pg1, result_30_im, UChi_00_im);\
    result_31_re = svadd_x(pg1, result_31_re, UChi_01_re);\
    result_31_im = svadd_x(pg1, result_31_im, UChi_01_im);\
    result_32_re = svadd_x(pg1, result_32_re, UChi_02_re);\
    result_32_im = svadd_x(pg1, result_32_im, UChi_02_im);

#define ZP_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svadd_x(pg1, result_20_re, UChi_00_im);\
    result_20_im = svsub_x(pg1, result_20_im, UChi_00_re);\
    result_21_re = svadd_x(pg1, result_21_re, UChi_01_im);\
    result_21_im = svsub_x(pg1, result_21_im, UChi_01_re);\
    result_22_re = svadd_x(pg1, result_22_re, UChi_02_im);\
    result_22_im = svsub_x(pg1, result_22_im, UChi_02_re);\
    result_30_re = svsub_x(pg1, result_30_re, UChi_10_im);\
    result_30_im = svadd_x(pg1, result_30_im, UChi_10_re);\
    result_31_re = svsub_x(pg1, result_31_re, UChi_11_im);\
    result_31_im = svadd_x(pg1, result_31_im, UChi_11_re);\
    result_32_re = svsub_x(pg1, result_32_re, UChi_12_im);\
    result_32_im = svadd_x(pg1, result_32_im, UChi_12_re);

#define ZM_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svsub_x(pg1, result_20_re, UChi_00_im);\
    result_20_im = svadd_x(pg1, result_20_im, UChi_00_re);\
    result_21_re = svsub_x(pg1, result_21_re, UChi_01_im);\
    result_21_im = svadd_x(pg1, result_21_im, UChi_01_re);\
    result_22_re = svsub_x(pg1, result_22_re, UChi_02_im);\
    result_22_im = svadd_x(pg1, result_22_im, UChi_02_re);\
    result_30_re = svadd_x(pg1, result_30_re, UChi_10_im);\
    result_30_im = svsub_x(pg1, result_30_im, UChi_10_re);\
    result_31_re = svadd_x(pg1, result_31_re, UChi_11_im);\
    result_31_im = svsub_x(pg1, result_31_im, UChi_11_re);\
    result_32_re = svadd_x(pg1, result_32_re, UChi_12_im);\
    result_32_im = svsub_x(pg1, result_32_im, UChi_12_re);

#define TP_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svadd_x(pg1, result_20_re, UChi_00_re);\
    result_20_im = svadd_x(pg1, result_20_im, UChi_00_im);\
    result_21_re = svadd_x(pg1, result_21_re, UChi_01_re);\
    result_21_im = svadd_x(pg1, result_21_im, UChi_01_im);\
    result_22_re = svadd_x(pg1, result_22_re, UChi_02_re);\
    result_22_im = svadd_x(pg1, result_22_im, UChi_02_im);\
    result_30_re = svadd_x(pg1, result_30_re, UChi_10_re);\
    result_30_im = svadd_x(pg1, result_30_im, UChi_10_im);\
    result_31_re = svadd_x(pg1, result_31_re, UChi_11_re);\
    result_31_im = svadd_x(pg1, result_31_im, UChi_11_im);\
    result_32_re = svadd_x(pg1, result_32_re, UChi_12_re);\
    result_32_im = svadd_x(pg1, result_32_im, UChi_12_im);

#define TM_RECON_ACCUM\
    result_00_re = svadd_x(pg1, result_00_re, UChi_00_re);\
    result_00_im = svadd_x(pg1, result_00_im, UChi_00_im);\
    result_01_re = svadd_x(pg1, result_01_re, UChi_01_re);\
    result_01_im = svadd_x(pg1, result_01_im, UChi_01_im);\
    result_02_re = svadd_x(pg1, result_02_re, UChi_02_re);\
    result_02_im = svadd_x(pg1, result_02_im, UChi_02_im);\
    result_10_re = svadd_x(pg1, result_10_re, UChi_10_re);\
    result_10_im = svadd_x(pg1, result_10_im, UChi_10_im);\
    result_11_re = svadd_x(pg1, result_11_re, UChi_11_re);\
    result_11_im = svadd_x(pg1, result_11_im, UChi_11_im);\
    result_12_re = svadd_x(pg1, result_12_re, UChi_12_re);\
    result_12_im = svadd_x(pg1, result_12_im, UChi_12_im);\
    result_20_re = svsub_x(pg1, result_20_re, UChi_00_re);\
    result_20_im = svsub_x(pg1, result_20_im, UChi_00_im);\
    result_21_re = svsub_x(pg1, result_21_re, UChi_01_re);\
    result_21_im = svsub_x(pg1, result_21_im, UChi_01_im);\
    result_22_re = svsub_x(pg1, result_22_re, UChi_02_re);\
    result_22_im = svsub_x(pg1, result_22_im, UChi_02_im);\
    result_30_re = svsub_x(pg1, result_30_re, UChi_10_re);\
    result_30_im = svsub_x(pg1, result_30_im, UChi_10_im);\
    result_31_re = svsub_x(pg1, result_31_re, UChi_11_re);\
    result_31_im = svsub_x(pg1, result_31_im, UChi_11_im);\
    result_32_re = svsub_x(pg1, result_32_re, UChi_12_re);\
    result_32_im = svsub_x(pg1, result_32_im, UChi_12_im);

#define HAND_STENCIL_LEG(PROJ,PERM,DIR,RECON)		\
  offset = nbr[ss*8+DIR];				\
  perm   = prm[ss*8+DIR];				\
  LOAD_CHIMU(PERM);					\
  PROJ;							\
  if (perm) {						\
    PERMUTE_DIR(PERM);					\
  }							\
  synchronise(); 					\
  MULT_2SPIN(DIR);					\
  RECON;

#define HAND_RESULT(ss)				\
  {	SiteSpinor & ref (out[ss]);	base = (uint64_t)ref;		\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)), result_00_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)), result_00_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)), result_01_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)), result_01_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)), result_02_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)), result_02_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)), result_10_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)), result_10_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1)), result_11_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 1 + 1)), result_11_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2)), result_12_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 2 + 1)), result_12_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0)), result_20_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 0 + 1)), result_20_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1)), result_21_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 1 + 1)), result_21_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2)), result_22_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 2 + 2 * 2 + 1)), result_22_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 0)), result_30_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 0 + 1)), result_30_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 1)), result_31_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 1 + 1)), result_31_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 2)), result_32_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 3 + 2 * 2 + 1)), result_32_im);\
  }

#define LOAD_T		\
  { const SiteSpinor & ref (in[0]);	base = (uint64_t)ref; \
    DEBUG1_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)));\
    DEBUG1_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)));\
    DEBUG2_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)));\
    DEBUG2_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)));\
    DEBUG3_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)));\
    DEBUG3_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)));\
    DEBUG4_re = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)));\
    DEBUG4_im = svld1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)));}

#define STORE_T				\
  {	SiteSpinor & ref (out[0]);	base = (uint64_t)ref;		\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)), DEBUG1_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)), DEBUG1_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)), DEBUG2_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)), DEBUG2_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)), DEBUG3_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)), DEBUG3_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)), DEBUG4_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)), DEBUG4_im);\
  }

#define PRINT_T				\
  {	SiteSpinor & ref (out[0]);	base = (uint64_t)ref;		\
    std::cout << ref[0][0] << std::endl;		\
    std::cout << ref[0][1] << std::endl;		\
    std::cout << ref[0][2] << std::endl;		\
    std::cout << ref[1][0] << std::endl;		\
  }

  #define STOREPRINT_T				\
    {	SiteSpinor & ref (out[0]);	base = (uint64_t)ref;		\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0)), DEBUG1_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 0 + 1)), DEBUG1_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1)), DEBUG2_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 1 + 1)), DEBUG2_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2)), DEBUG3_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 0 + 2 * 2 + 1)), DEBUG3_im);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0)), DEBUG4_re);\
    svst1(pg1, (float64_t*)(base + 64 * (2 * 3 * 1 + 2 * 0 + 1)), DEBUG4_im);\
      std::cout << "XX " << ref[0][0] << std::endl;		\
      std::cout << "XX " << ref[0][1] << std::endl;		\
      std::cout << "XX " << ref[0][2] << std::endl;		\
      std::cout << "XX " << ref[1][0] << std::endl;		\
    }


#define HAND_DECLARATIONS(Simd)			\
    svfloat64_t result_00_re;\
    svfloat64_t result_00_im;\
    svfloat64_t result_01_re;\
    svfloat64_t result_01_im;\
    svfloat64_t result_02_re;\
    svfloat64_t result_02_im;\
    svfloat64_t result_10_re;\
    svfloat64_t result_10_im;\
    svfloat64_t result_11_re;\
    svfloat64_t result_11_im;\
    svfloat64_t result_12_re;\
    svfloat64_t result_12_im;\
    svfloat64_t result_20_re;\
    svfloat64_t result_20_im;\
    svfloat64_t result_21_re;\
    svfloat64_t result_21_im;\
    svfloat64_t result_22_re;\
    svfloat64_t result_22_im;\
    svfloat64_t result_30_re;\
    svfloat64_t result_30_im;\
    svfloat64_t result_31_re;\
    svfloat64_t result_31_im;\
    svfloat64_t result_32_re;\
    svfloat64_t result_32_im;\
    svfloat64_t Chi_00_re;\
    svfloat64_t Chi_00_im;\
    svfloat64_t Chi_01_re;\
    svfloat64_t Chi_01_im;\
    svfloat64_t Chi_02_re;\
    svfloat64_t Chi_02_im;\
    svfloat64_t Chi_10_re;\
    svfloat64_t Chi_10_im;\
    svfloat64_t Chi_11_re;\
    svfloat64_t Chi_11_im;\
    svfloat64_t Chi_12_re;\
    svfloat64_t Chi_12_im;\
    svfloat64_t UChi_00_re;\
    svfloat64_t UChi_00_im;\
    svfloat64_t UChi_01_re;\
    svfloat64_t UChi_01_im;\
    svfloat64_t UChi_02_re;\
    svfloat64_t UChi_02_im;\
    svfloat64_t UChi_10_re;\
    svfloat64_t UChi_10_im;\
    svfloat64_t UChi_11_re;\
    svfloat64_t UChi_11_im;\
    svfloat64_t UChi_12_re;\
    svfloat64_t UChi_12_im;\
    svfloat64_t U_00_re;\
    svfloat64_t U_00_im;\
    svfloat64_t U_10_re;\
    svfloat64_t U_10_im;\
    svfloat64_t U_20_re;\
    svfloat64_t U_20_im;\
    svfloat64_t U_01_re;\
    svfloat64_t U_01_im;\
    svfloat64_t U_11_re;\
    svfloat64_t U_11_im;\
    svfloat64_t U_21_re;\
    svfloat64_t U_21_im;\
    svfloat64_t Chimu_00_re;\
    svfloat64_t Chimu_00_im;\
    svfloat64_t Chimu_01_re;\
    svfloat64_t Chimu_01_im;\
    svfloat64_t Chimu_02_re;\
    svfloat64_t Chimu_02_im;\
    svfloat64_t Chimu_10_re;\
    svfloat64_t Chimu_10_im;\
    svfloat64_t Chimu_11_re;\
    svfloat64_t Chimu_11_im;\
    svfloat64_t Chimu_12_re;\
    svfloat64_t Chimu_12_im;\
    svfloat64_t Chimu_20_re;\
    svfloat64_t Chimu_20_im;\
    svfloat64_t Chimu_21_re;\
    svfloat64_t Chimu_21_im;\
    svfloat64_t Chimu_22_re;\
    svfloat64_t Chimu_22_im;\
    svfloat64_t Chimu_30_re;\
    svfloat64_t Chimu_30_im;\
    svfloat64_t Chimu_31_re;\
    svfloat64_t Chimu_31_im;\
    svfloat64_t Chimu_32_re;\
    svfloat64_t Chimu_32_im;\
    svfloat64_t DEBUG1_re;\
    svfloat64_t DEBUG1_im;\
    svfloat64_t DEBUG2_re;\
    svfloat64_t DEBUG2_im;\
    svfloat64_t DEBUG3_re;\
    svfloat64_t DEBUG3_im;\
    svfloat64_t DEBUG4_re;\
    svfloat64_t DEBUG4_im;\
  svbool_t pg1 = svptrue_b64();




/*
#define Chimu_00 Chi_00
#define Chimu_01 Chi_01
#define Chimu_02 Chi_02
#define Chimu_10 Chi_10
#define Chimu_11 Chi_11
#define Chimu_12 Chi_12
#define Chimu_20 UChi_00
#define Chimu_21 UChi_01
#define Chimu_22 UChi_02
#define Chimu_30 UChi_10
#define Chimu_31 UChi_11
#define Chimu_32 UChi_12
*/

#ifndef GRID_SYCL
#define GRID_OMP_THREAD
#endif

#ifdef GRID_OMP_THREAD
template<class SimdVec>
double dslash_kernel_cpu(int nrep,SimdVec *Up,SimdVec *outp,SimdVec *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm)
{
  typedef  std::chrono::system_clock          Clock;
  typedef  std::chrono::time_point<Clock> TimePoint;
  typedef  std::chrono::microseconds          Usecs;

  typedef SimdVec Simd;
  typedef Simd SiteSpinor[4][3];
  typedef Simd SiteHalfSpinor[2][3];
  typedef Simd SiteDoubledGaugeField[8][3][3];

  //  typedef typename Simd::scalar_type S;
  //  typedef typename Simd::vector_type V;

  SiteSpinor *out = (SiteSpinor *) outp;
  SiteSpinor *in  = (SiteSpinor *) inp;
  SiteDoubledGaugeField *U  = (SiteDoubledGaugeField *) Up;

  //  Simd complex_i;  vsplat(complex_i, S(0.0, 1.0));

  TimePoint start; double usec;
  for(int rep=0;rep<nrep;rep++){
    if ( rep==1 ) start = Clock::now();
    //    static_assert(std::is_trivially_constructible<Simd>::value," SIMD is not trivial constructible");
    //    static_assert(std::is_trivially_constructible<SiteSpinor>::value," not trivial constructible");
    //    static_assert(std::is_trivially_constructible<SiteDoubledGaugeField>::value," not trivial constructible");
    //    static_assert(std::is_trivially_default_constructible<Simd>::value," SIMD is not trivial default constructible");
    //    static_assert(std::is_trivially_copyable<Simd>::value," SIMD is not copy constructible");
    //    static_assert(std::is_trivially_copyable<sycl::vec<double,4> >::value," sycl::vec is trivially copy constructible");
#ifdef OMP
//#define OMP5
  #ifdef OMP5
  #warning "OpenMP 5.0 target pragma"
  #pragma omp target map(in[0:nsite*Ls], out[0:nsite*Ls],U[0:nsite],nbr[0:nsite*8*Ls],prm[0:nsite*8*Ls])
  #pragma omp teams distribute parallel for
  #else
  #pragma omp parallel for
  #endif
#endif

    // Debug

    HAND_DECLARATIONS(Simd);
    int mylane=0;
    int offset=0,perm;
    uint64_t base;
    uint64_t ssite=0;
    uint64_t sU = ssite;
    uint64_t ss = sU*Ls;

    LOAD_T;

    std::cout << "XX Load/store" << std::endl;
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG4" << std::endl;
    DEBUG1_re = DEBUG4_re;\
    DEBUG1_im = DEBUG4_im;
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2*DEBUG3" << std::endl;
    DEBUG1_re = svmul_x(pg1, DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = svmul_x(pg1, DEBUG2_re, DEBUG3_im);\
    DEBUG1_re = svmls_x(pg1, DEBUG1_re, DEBUG2_im, DEBUG3_im);\
    DEBUG1_im = svmla_x(pg1, DEBUG1_im, DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += DEBUG2*DEBUG3" << std::endl;
    DEBUG1_re = svmla_x(pg1, DEBUG1_re, DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = svmla_x(pg1, DEBUG1_im, DEBUG2_re, DEBUG3_im);\
    DEBUG1_re = svmls_x(pg1, DEBUG1_re, DEBUG2_im, DEBUG3_im);\
    DEBUG1_im = svmla_x(pg1, DEBUG1_im, DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2-DEBUG3" << std::endl;
    DEBUG1_re = svsub_x(pg1, DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = svsub_x(pg1, DEBUG2_im, DEBUG3_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 -= DEBUG2" << std::endl;
    DEBUG1_re = svsub_x(pg1, DEBUG1_re, DEBUG2_re);\
    DEBUG1_im = svsub_x(pg1, DEBUG1_im, DEBUG2_im);
    STOREPRINT_T;
        
    std::cout << "XX DEBUG1 = DEBUG2+DEBUG3" << std::endl;
    DEBUG1_re = svadd_x(pg1, DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = svadd_x(pg1, DEBUG2_im, DEBUG3_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += DEBUG2" << std::endl;
    DEBUG1_re = svadd_x(pg1, DEBUG1_re, DEBUG2_re);\
    DEBUG1_im = svadd_x(pg1, DEBUG1_im, DEBUG2_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = timesI(DEBUG2)" << std::endl;
    DEBUG1_re = svneg_x(pg1, DEBUG2_im);\
    DEBUG1_im = DEBUG2_re;
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2+timesI(DEBUG3)" << std::endl;
    DEBUG1_re = svsub_x(pg1, DEBUG2_re, DEBUG3_im);\
    DEBUG1_im = svadd_x(pg1, DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2-timesI(DEBUG3)" << std::endl;
    DEBUG1_re = svadd_x(pg1, DEBUG2_re, DEBUG3_im);\
    DEBUG1_im = svsub_x(pg1, DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += timesI(DEBUG2)" << std::endl;
    DEBUG1_re = svsub_x(pg1, DEBUG1_re, DEBUG2_im);\
    DEBUG1_im = svadd_x(pg1, DEBUG1_im, DEBUG2_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 -= timesI(DEBUG2)" << std::endl;
    DEBUG1_re = svadd_x(pg1, DEBUG1_re, DEBUG2_im);\
    DEBUG1_im = svsub_x(pg1, DEBUG1_im, DEBUG2_re);
    STOREPRINT_T;

    std::cout << "XX " << std::endl;
    DEBUG1_re = DEBUG2_im;\
    DEBUG1_im = svneg_x(pg1, DEBUG2_re);
    STOREPRINT_T;

    std::cout << "XX " << std::endl;
    STOREPRINT_T;

    std::cout << "XX " << std::endl;
    STOREPRINT_T;

    break; // exit loop
  }

  Usecs elapsed =std::chrono::duration_cast<Usecs>(Clock::now()-start);
  usec = elapsed.count();
  return usec;
}
#endif

#ifdef GRID_SYCL
#include <CL/sycl.hpp>

// Sycl loop
template<class SimdVec>
double dslash_kernel_cpu(int nrep,SimdVec *Up,SimdVec *outp,SimdVec *inp,uint64_t *nbrp,uint64_t nsite,uint64_t Ls,uint8_t *prmp)
{
  typedef  std::chrono::system_clock          Clock;
  typedef  std::chrono::time_point<Clock> TimePoint;
  typedef  std::chrono::microseconds          Usecs;
  TimePoint start; double usec;

  using namespace cl::sycl;

  const uint64_t    NN = nsite*Ls;

  typedef typename SimdVec::vector_type::word_type Float;
  const uint64_t    Nsimd = SimdVec::Nsimd();

  assert(Nsimd==sizeof(SimdVec)/sizeof(Float)/2);
  uint64_t begin=0;
  uint64_t end  =NN;

  {
  const uint64_t _umax=nsite*9*8;
  const uint64_t _fmax=nsite*12*Ls;
  const uint64_t _nbrmax=nsite*Ls*8;
  cl::sycl::range<1> num{NN};
  cl::sycl::range<1> umax{_umax};
  cl::sycl::range<1> fmax{_fmax};
  cl::sycl::range<1> nbrmax{_nbrmax};

    // Create queue
#if GRID_SYCL_GPU
  auto sel =cl::sycl::gpu_selector();
#else
  auto sel =cl::sycl::cpu_selector();
#endif
  cl::sycl::queue q(sel);

#ifdef SVM
  SimdVec * Usvm  =(SimdVec *) malloc_shared(_umax*sizeof(SimdVec),q);
  SimdVec * insvm =(SimdVec *) malloc_shared(_fmax*sizeof(SimdVec),q);
  SimdVec * outsvm=(SimdVec *) malloc_shared(_fmax*sizeof(SimdVec),q);
  uint64_t* nbrsvm=(uint64_t *) malloc_shared(_nbrmax*sizeof(uint64_t),q);
  uint8_t * prmsvm=(uint8_t  *) malloc_shared(_nbrmax*sizeof(uint8_t),q);
  std::cout << "SVM allocated arrays for SIMD "<<Nsimd <<std::endl;
  for(uint64_t n=0;n<_umax;n++) Usvm[n] = Up[n];
  for(uint64_t n=0;n<_fmax;n++) insvm[n] = inp[n];
  for(uint64_t n=0;n<_nbrmax;n++) nbrsvm[n] = nbrp[n];
  for(uint64_t n=0;n<_nbrmax;n++) prmsvm[n] = prmp[n];
  std::cout << "SVM assigned arrays" <<std::endl;
#else
  cl::sycl::buffer<SimdVec,1>     Up_b   { &  Up[begin],umax};
  cl::sycl::buffer<SimdVec,1>     inp_b  { & inp[begin],fmax};
  cl::sycl::buffer<SimdVec,1>     outp_b { &outp[begin],fmax};
  cl::sycl::buffer<uint64_t,1> nbr_b  { &nbrp[begin],nbrmax};
  cl::sycl::buffer<uint8_t,1>  prm_b  { &prmp[begin],nbrmax};
#endif

  for(int rep=0;rep<nrep;rep++) {
    if ( rep==1 ) start = Clock::now();

    q.submit([&](handler &cgh) {
        // In the kernel A and B are read, but C is written
#ifndef SVM
        auto Up_k   =  Up_b.template get_access<access::mode::read>(cgh);
        auto inp_k  = inp_b.template get_access<access::mode::read>(cgh);
        auto nbr_k  = nbr_b.template get_access<access::mode::read>(cgh);
        auto prm_k  = prm_b.template get_access<access::mode::read>(cgh);
	auto outp_k =outp_b.template get_access<access::mode::read_write>(cgh);
#endif
	typedef decltype(coalescedRead(inp[0],0)) Simd;
	typedef SimdVec SiteSpinor[4][3];
	typedef SimdVec SiteHalfSpinor[2][3];
	typedef SimdVec SiteDoubledGaugeField[8][3][3];

#ifdef GRID_SYCL_SIMT
	//	cl::sycl::range<3> global{Nsimd,Ls,nsite};
	//	cl::sycl::range<3> local {Nsimd,1,1};
	cl::sycl::range<3> global{nsite,Ls,Nsimd};
	cl::sycl::range<3> local {1,1,Nsimd};
#else
	//	cl::sycl::range<3> global{1,Ls,nsite};
	//	cl::sycl::range<3> local {1,Ls,1};
	cl::sycl::range<3> global{nsite,Ls,1};
	cl::sycl::range<3> local {1,Ls,1};
#endif
	cgh.parallel_for<class dslash>(cl::sycl::nd_range<3>(global,local),
				       [=] (cl::sycl::nd_item<3> item)
				       [[cl::intel_reqd_sub_group_size(sizeof(SimdVec)/sizeof(Float)/2)]]
	{
 	    auto sg = item.get_sub_group();

	    auto mylane = item.get_global_id(2);
	    auto    s   = item.get_global_id(1);
	    auto   sU   = item.get_global_id(0);
	    auto    ss  = s+Ls*sU;

	    HAND_DECLARATIONS(Simd);
#ifdef SVM
	    SiteDoubledGaugeField *U  = (SiteDoubledGaugeField *) Usvm;
	    SiteSpinor *out = (SiteSpinor *) outsvm;
	    SiteSpinor *in  = (SiteSpinor *) insvm;
	    uint64_t *nbr   = (uint64_t *) nbrsvm;
	    uint8_t *prm    = (uint8_t  *) prmsvm;
#else
	    SiteDoubledGaugeField *U  = (SiteDoubledGaugeField *) Up_k.get_pointer().get();
	    SiteSpinor *out = (SiteSpinor *) outp_k.get_pointer().get();
	    SiteSpinor *in  = (SiteSpinor *) inp_k.get_pointer().get();
	    uint64_t *nbr   = (uint64_t *) nbr_k.get_pointer().get();
	    uint8_t *prm    = (uint8_t  *) prm_k.get_pointer().get();
#endif

	    int offset,perm;
	    HAND_STENCIL_LEG(XM_PROJ,3,Xp,XM_RECON);
	    HAND_STENCIL_LEG(YM_PROJ,2,Yp,YM_RECON_ACCUM);
	    HAND_STENCIL_LEG(ZM_PROJ,1,Zp,ZM_RECON_ACCUM);
	    HAND_STENCIL_LEG(TM_PROJ,0,Tp,TM_RECON_ACCUM);
	    HAND_STENCIL_LEG(XP_PROJ,3,Xm,XP_RECON_ACCUM);
	    HAND_STENCIL_LEG(YP_PROJ,2,Ym,YP_RECON_ACCUM);
	    HAND_STENCIL_LEG(ZP_PROJ,1,Zm,ZP_RECON_ACCUM);
	    HAND_STENCIL_LEG(TP_PROJ,0,Tm,TP_RECON_ACCUM);
	    HAND_RESULT(ss);

	  });

   }); //< End of our commands for this queue
  }

  q.wait();
  //< Buffer outp_b goes out of scope and copies back values to outp
  // Queue out of scope, waits
#ifdef SVM
  for(int n=0;n<_fmax;n++) outp[n] = outsvm[n];

  free(Usvm,q);
  free(insvm,q);
  free(outsvm,q);
  free(nbrsvm,q);
  free(prmsvm,q);
#endif
  }
  Usecs elapsed =std::chrono::duration_cast<Usecs>(Clock::now()-start);
  usec = elapsed.count();
  return usec;
}
#endif
