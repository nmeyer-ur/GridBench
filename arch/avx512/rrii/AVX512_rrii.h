// intrinsify
// arch                AVX512
// vector length       64 bytes, 512 bits
// header file         immintrin.h
// float type          double
// float* typecast     (double*)
// simd type           __m512d

#include <immintrin.h>
    
#include <stdio.h>
#include <immintrin.h>

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
    Chimu_00_re = _mm512_load_pd((double*)(base + 64 * 0));\
    Chimu_00_im = _mm512_load_pd((double*)(base + 64 * 1));\
    Chimu_01_re = _mm512_load_pd((double*)(base + 64 * 2));\
    Chimu_01_im = _mm512_load_pd((double*)(base + 64 * 3));\
    Chimu_02_re = _mm512_load_pd((double*)(base + 64 * 4));\
    Chimu_02_im = _mm512_load_pd((double*)(base + 64 * 5));\
    Chimu_10_re = _mm512_load_pd((double*)(base + 64 * 6));\
    Chimu_10_im = _mm512_load_pd((double*)(base + 64 * 7));\
    Chimu_11_re = _mm512_load_pd((double*)(base + 64 * 8));\
    Chimu_11_im = _mm512_load_pd((double*)(base + 64 * 9));\
    Chimu_12_re = _mm512_load_pd((double*)(base + 64 * 10));\
    Chimu_12_im = _mm512_load_pd((double*)(base + 64 * 11));\
    Chimu_20_re = _mm512_load_pd((double*)(base + 64 * 12));\
    Chimu_20_im = _mm512_load_pd((double*)(base + 64 * 13));\
    Chimu_21_re = _mm512_load_pd((double*)(base + 64 * 14));\
    Chimu_21_im = _mm512_load_pd((double*)(base + 64 * 15));\
    Chimu_22_re = _mm512_load_pd((double*)(base + 64 * 16));\
    Chimu_22_im = _mm512_load_pd((double*)(base + 64 * 17));\
    Chimu_30_re = _mm512_load_pd((double*)(base + 64 * 18));\
    Chimu_30_im = _mm512_load_pd((double*)(base + 64 * 19));\
    Chimu_31_re = _mm512_load_pd((double*)(base + 64 * 20));\
    Chimu_31_im = _mm512_load_pd((double*)(base + 64 * 21));\
    Chimu_32_re = _mm512_load_pd((double*)(base + 64 * 22));\
    Chimu_32_im = _mm512_load_pd((double*)(base + 64 * 23));}

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
    U_00_re = _mm512_load_pd((double*)(base + 64 * 0));\
    U_00_im = _mm512_load_pd((double*)(base + 64 * 1));\
    U_10_re = _mm512_load_pd((double*)(base + 64 * 6));\
    U_10_im = _mm512_load_pd((double*)(base + 64 * 7));\
    U_20_re = _mm512_load_pd((double*)(base + 64 * 12));\
    U_20_im = _mm512_load_pd((double*)(base + 64 * 13));\
    U_01_re = _mm512_load_pd((double*)(base + 64 * 2));\
    U_01_im = _mm512_load_pd((double*)(base + 64 * 3));\
    U_11_re = _mm512_load_pd((double*)(base + 64 * 8));\
    U_11_im = _mm512_load_pd((double*)(base + 64 * 9));\
    U_21_re = _mm512_load_pd((double*)(base + 64 * 14));\
    U_21_im = _mm512_load_pd((double*)(base + 64 * 15));\
    UChi_00_re = _mm512_mul_pd(U_00_re, Chi_00_re);\
    UChi_00_im = _mm512_mul_pd(U_00_re, Chi_00_im);\
    UChi_00_re = _mm512_fnmadd_pd(U_00_im, Chi_00_im, UChi_00_re);\
    UChi_00_im = _mm512_fmadd_pd(U_00_im, Chi_00_re, UChi_00_im);\
    UChi_10_re = _mm512_mul_pd(U_00_re, Chi_10_re);\
    UChi_10_im = _mm512_mul_pd(U_00_re, Chi_10_im);\
    UChi_10_re = _mm512_fnmadd_pd(U_00_im, Chi_10_im, UChi_10_re);\
    UChi_10_im = _mm512_fmadd_pd(U_00_im, Chi_10_re, UChi_10_im);\
    UChi_01_re = _mm512_mul_pd(U_10_re, Chi_00_re);\
    UChi_01_im = _mm512_mul_pd(U_10_re, Chi_00_im);\
    UChi_01_re = _mm512_fnmadd_pd(U_10_im, Chi_00_im, UChi_01_re);\
    UChi_01_im = _mm512_fmadd_pd(U_10_im, Chi_00_re, UChi_01_im);\
    UChi_11_re = _mm512_mul_pd(U_10_re, Chi_10_re);\
    UChi_11_im = _mm512_mul_pd(U_10_re, Chi_10_im);\
    UChi_11_re = _mm512_fnmadd_pd(U_10_im, Chi_10_im, UChi_11_re);\
    UChi_11_im = _mm512_fmadd_pd(U_10_im, Chi_10_re, UChi_11_im);\
    UChi_02_re = _mm512_mul_pd(U_20_re, Chi_00_re);\
    UChi_02_im = _mm512_mul_pd(U_20_re, Chi_00_im);\
    UChi_02_re = _mm512_fnmadd_pd(U_20_im, Chi_00_im, UChi_02_re);\
    UChi_02_im = _mm512_fmadd_pd(U_20_im, Chi_00_re, UChi_02_im);\
    UChi_12_re = _mm512_mul_pd(U_20_re, Chi_10_re);\
    UChi_12_im = _mm512_mul_pd(U_20_re, Chi_10_im);\
    UChi_12_re = _mm512_fnmadd_pd(U_20_im, Chi_10_im, UChi_12_re);\
    UChi_12_im = _mm512_fmadd_pd(U_20_im, Chi_10_re, UChi_12_im);\
    UChi_00_re = _mm512_fmadd_pd(U_01_re, Chi_01_re, UChi_00_re);\
    UChi_00_im = _mm512_fmadd_pd(U_01_re, Chi_01_im, UChi_00_im);\
    UChi_00_re = _mm512_fnmadd_pd(U_01_im, Chi_01_im, UChi_00_re);\
    UChi_00_im = _mm512_fmadd_pd(U_01_im, Chi_01_re, UChi_00_im);\
    UChi_10_re = _mm512_fmadd_pd(U_01_re, Chi_11_re, UChi_10_re);\
    UChi_10_im = _mm512_fmadd_pd(U_01_re, Chi_11_im, UChi_10_im);\
    UChi_10_re = _mm512_fnmadd_pd(U_01_im, Chi_11_im, UChi_10_re);\
    UChi_10_im = _mm512_fmadd_pd(U_01_im, Chi_11_re, UChi_10_im);\
    UChi_01_re = _mm512_fmadd_pd(U_11_re, Chi_01_re, UChi_01_re);\
    UChi_01_im = _mm512_fmadd_pd(U_11_re, Chi_01_im, UChi_01_im);\
    UChi_01_re = _mm512_fnmadd_pd(U_11_im, Chi_01_im, UChi_01_re);\
    UChi_01_im = _mm512_fmadd_pd(U_11_im, Chi_01_re, UChi_01_im);\
    UChi_11_re = _mm512_fmadd_pd(U_11_re, Chi_11_re, UChi_11_re);\
    UChi_11_im = _mm512_fmadd_pd(U_11_re, Chi_11_im, UChi_11_im);\
    UChi_11_re = _mm512_fnmadd_pd(U_11_im, Chi_11_im, UChi_11_re);\
    UChi_11_im = _mm512_fmadd_pd(U_11_im, Chi_11_re, UChi_11_im);\
    UChi_02_re = _mm512_fmadd_pd(U_21_re, Chi_01_re, UChi_02_re);\
    UChi_02_im = _mm512_fmadd_pd(U_21_re, Chi_01_im, UChi_02_im);\
    UChi_02_re = _mm512_fnmadd_pd(U_21_im, Chi_01_im, UChi_02_re);\
    UChi_02_im = _mm512_fmadd_pd(U_21_im, Chi_01_re, UChi_02_im);\
    UChi_12_re = _mm512_fmadd_pd(U_21_re, Chi_11_re, UChi_12_re);\
    UChi_12_im = _mm512_fmadd_pd(U_21_re, Chi_11_im, UChi_12_im);\
    UChi_12_re = _mm512_fnmadd_pd(U_21_im, Chi_11_im, UChi_12_re);\
    UChi_12_im = _mm512_fmadd_pd(U_21_im, Chi_11_re, UChi_12_im);\
    U_00_re = _mm512_load_pd((double*)(base + 64 * 4));\
    U_00_im = _mm512_load_pd((double*)(base + 64 * 5));\
    U_10_re = _mm512_load_pd((double*)(base + 64 * 10));\
    U_10_im = _mm512_load_pd((double*)(base + 64 * 11));\
    U_20_re = _mm512_load_pd((double*)(base + 64 * 16));\
    U_20_im = _mm512_load_pd((double*)(base + 64 * 17));\
    UChi_00_re = _mm512_fmadd_pd(U_00_re, Chi_02_re, UChi_00_re);\
    UChi_00_im = _mm512_fmadd_pd(U_00_re, Chi_02_im, UChi_00_im);\
    UChi_00_re = _mm512_fnmadd_pd(U_00_im, Chi_02_im, UChi_00_re);\
    UChi_00_im = _mm512_fmadd_pd(U_00_im, Chi_02_re, UChi_00_im);\
    UChi_10_re = _mm512_fmadd_pd(U_00_re, Chi_12_re, UChi_10_re);\
    UChi_10_im = _mm512_fmadd_pd(U_00_re, Chi_12_im, UChi_10_im);\
    UChi_10_re = _mm512_fnmadd_pd(U_00_im, Chi_12_im, UChi_10_re);\
    UChi_10_im = _mm512_fmadd_pd(U_00_im, Chi_12_re, UChi_10_im);\
    UChi_01_re = _mm512_fmadd_pd(U_10_re, Chi_02_re, UChi_01_re);\
    UChi_01_im = _mm512_fmadd_pd(U_10_re, Chi_02_im, UChi_01_im);\
    UChi_01_re = _mm512_fnmadd_pd(U_10_im, Chi_02_im, UChi_01_re);\
    UChi_01_im = _mm512_fmadd_pd(U_10_im, Chi_02_re, UChi_01_im);\
    UChi_11_re = _mm512_fmadd_pd(U_10_re, Chi_12_re, UChi_11_re);\
    UChi_11_im = _mm512_fmadd_pd(U_10_re, Chi_12_im, UChi_11_im);\
    UChi_11_re = _mm512_fnmadd_pd(U_10_im, Chi_12_im, UChi_11_re);\
    UChi_11_im = _mm512_fmadd_pd(U_10_im, Chi_12_re, UChi_11_im);\
    UChi_02_re = _mm512_fmadd_pd(U_20_re, Chi_02_re, UChi_02_re);\
    UChi_02_im = _mm512_fmadd_pd(U_20_re, Chi_02_im, UChi_02_im);\
    UChi_02_re = _mm512_fnmadd_pd(U_20_im, Chi_02_im, UChi_02_re);\
    UChi_02_im = _mm512_fmadd_pd(U_20_im, Chi_02_re, UChi_02_im);\
    UChi_12_re = _mm512_fmadd_pd(U_20_re, Chi_12_re, UChi_12_re);\
    UChi_12_im = _mm512_fmadd_pd(U_20_re, Chi_12_im, UChi_12_im);\
    UChi_12_re = _mm512_fnmadd_pd(U_20_im, Chi_12_im, UChi_12_re);\
    UChi_12_im = _mm512_fmadd_pd(U_20_im, Chi_12_re, UChi_12_im);}

//      hspin(0)=fspin(0)+timesI(fspin(3));
//      hspin(1)=fspin(1)+timesI(fspin(2));
#define XP_PROJ \
    Chi_00_re = _mm512_sub_pd(Chimu_00_re, Chimu_30_im);\
    Chi_00_im = _mm512_add_pd(Chimu_00_im, Chimu_30_re);\
    Chi_01_re = _mm512_sub_pd(Chimu_01_re, Chimu_31_im);\
    Chi_01_im = _mm512_add_pd(Chimu_01_im, Chimu_31_re);\
    Chi_02_re = _mm512_sub_pd(Chimu_02_re, Chimu_32_im);\
    Chi_02_im = _mm512_add_pd(Chimu_02_im, Chimu_32_re);\
    Chi_10_re = _mm512_sub_pd(Chimu_10_re, Chimu_20_im);\
    Chi_10_im = _mm512_add_pd(Chimu_10_im, Chimu_20_re);\
    Chi_11_re = _mm512_sub_pd(Chimu_11_re, Chimu_21_im);\
    Chi_11_im = _mm512_add_pd(Chimu_11_im, Chimu_21_re);\
    Chi_12_re = _mm512_sub_pd(Chimu_12_re, Chimu_22_im);\
    Chi_12_im = _mm512_add_pd(Chimu_12_im, Chimu_22_re);

#define YP_PROJ \
    Chi_00_re = _mm512_sub_pd(Chimu_00_re, Chimu_30_re);\
    Chi_00_im = _mm512_sub_pd(Chimu_00_im, Chimu_30_im);\
    Chi_01_re = _mm512_sub_pd(Chimu_01_re, Chimu_31_re);\
    Chi_01_im = _mm512_sub_pd(Chimu_01_im, Chimu_31_im);\
    Chi_02_re = _mm512_sub_pd(Chimu_02_re, Chimu_32_re);\
    Chi_02_im = _mm512_sub_pd(Chimu_02_im, Chimu_32_im);\
    Chi_10_re = _mm512_add_pd(Chimu_10_re, Chimu_20_re);\
    Chi_10_im = _mm512_add_pd(Chimu_10_im, Chimu_20_im);\
    Chi_11_re = _mm512_add_pd(Chimu_11_re, Chimu_21_re);\
    Chi_11_im = _mm512_add_pd(Chimu_11_im, Chimu_21_im);\
    Chi_12_re = _mm512_add_pd(Chimu_12_re, Chimu_22_re);\
    Chi_12_im = _mm512_add_pd(Chimu_12_im, Chimu_22_im);

#define ZP_PROJ \
    Chi_00_re = _mm512_sub_pd(Chimu_00_re, Chimu_20_im);\
    Chi_00_im = _mm512_add_pd(Chimu_00_im, Chimu_20_re);\
    Chi_01_re = _mm512_sub_pd(Chimu_01_re, Chimu_21_im);\
    Chi_01_im = _mm512_add_pd(Chimu_01_im, Chimu_21_re);\
    Chi_02_re = _mm512_sub_pd(Chimu_02_re, Chimu_22_im);\
    Chi_02_im = _mm512_add_pd(Chimu_02_im, Chimu_22_re);\
    Chi_10_re = _mm512_add_pd(Chimu_10_re, Chimu_30_im);\
    Chi_10_im = _mm512_sub_pd(Chimu_10_im, Chimu_30_re);\
    Chi_11_re = _mm512_add_pd(Chimu_11_re, Chimu_31_im);\
    Chi_11_im = _mm512_sub_pd(Chimu_11_im, Chimu_31_re);\
    Chi_12_re = _mm512_add_pd(Chimu_12_re, Chimu_32_im);\
    Chi_12_im = _mm512_sub_pd(Chimu_12_im, Chimu_32_re);

#define TP_PROJ \
    Chi_00_re = _mm512_add_pd(Chimu_00_re, Chimu_20_re);\
    Chi_00_im = _mm512_add_pd(Chimu_00_im, Chimu_20_im);\
    Chi_01_re = _mm512_add_pd(Chimu_01_re, Chimu_21_re);\
    Chi_01_im = _mm512_add_pd(Chimu_01_im, Chimu_21_im);\
    Chi_02_re = _mm512_add_pd(Chimu_02_re, Chimu_22_re);\
    Chi_02_im = _mm512_add_pd(Chimu_02_im, Chimu_22_im);\
    Chi_10_re = _mm512_add_pd(Chimu_10_re, Chimu_30_re);\
    Chi_10_im = _mm512_add_pd(Chimu_10_im, Chimu_30_im);\
    Chi_11_re = _mm512_add_pd(Chimu_11_re, Chimu_31_re);\
    Chi_11_im = _mm512_add_pd(Chimu_11_im, Chimu_31_im);\
    Chi_12_re = _mm512_add_pd(Chimu_12_re, Chimu_32_re);\
    Chi_12_im = _mm512_add_pd(Chimu_12_im, Chimu_32_im);


//      hspin(0)=fspin(0)-timesI(fspin(3));
//      hspin(1)=fspin(1)-timesI(fspin(2));
#define XM_PROJ \
    Chi_00_re = _mm512_add_pd(Chimu_00_re, Chimu_30_im);\
    Chi_00_im = _mm512_sub_pd(Chimu_00_im, Chimu_30_re);\
    Chi_01_re = _mm512_add_pd(Chimu_01_re, Chimu_31_im);\
    Chi_01_im = _mm512_sub_pd(Chimu_01_im, Chimu_31_re);\
    Chi_02_re = _mm512_add_pd(Chimu_02_re, Chimu_32_im);\
    Chi_02_im = _mm512_sub_pd(Chimu_02_im, Chimu_32_re);\
    Chi_10_re = _mm512_add_pd(Chimu_10_re, Chimu_20_im);\
    Chi_10_im = _mm512_sub_pd(Chimu_10_im, Chimu_20_re);\
    Chi_11_re = _mm512_add_pd(Chimu_11_re, Chimu_21_im);\
    Chi_11_im = _mm512_sub_pd(Chimu_11_im, Chimu_21_re);\
    Chi_12_re = _mm512_add_pd(Chimu_12_re, Chimu_22_im);\
    Chi_12_im = _mm512_sub_pd(Chimu_12_im, Chimu_22_re);

#define YM_PROJ \
    Chi_00_re = _mm512_add_pd(Chimu_00_re, Chimu_30_re);\
    Chi_00_im = _mm512_add_pd(Chimu_00_im, Chimu_30_im);\
    Chi_01_re = _mm512_add_pd(Chimu_01_re, Chimu_31_re);\
    Chi_01_im = _mm512_add_pd(Chimu_01_im, Chimu_31_im);\
    Chi_02_re = _mm512_add_pd(Chimu_02_re, Chimu_32_re);\
    Chi_02_im = _mm512_add_pd(Chimu_02_im, Chimu_32_im);\
    Chi_10_re = _mm512_sub_pd(Chimu_10_re, Chimu_20_re);\
    Chi_10_im = _mm512_sub_pd(Chimu_10_im, Chimu_20_im);\
    Chi_11_re = _mm512_sub_pd(Chimu_11_re, Chimu_21_re);\
    Chi_11_im = _mm512_sub_pd(Chimu_11_im, Chimu_21_im);\
    Chi_12_re = _mm512_sub_pd(Chimu_12_re, Chimu_22_re);\
    Chi_12_im = _mm512_sub_pd(Chimu_12_im, Chimu_22_im);

#define ZM_PROJ \
    Chi_00_re = _mm512_add_pd(Chimu_00_re, Chimu_20_im);\
    Chi_00_im = _mm512_sub_pd(Chimu_00_im, Chimu_20_re);\
    Chi_01_re = _mm512_add_pd(Chimu_01_re, Chimu_21_im);\
    Chi_01_im = _mm512_sub_pd(Chimu_01_im, Chimu_21_re);\
    Chi_02_re = _mm512_add_pd(Chimu_02_re, Chimu_22_im);\
    Chi_02_im = _mm512_sub_pd(Chimu_02_im, Chimu_22_re);\
    Chi_10_re = _mm512_sub_pd(Chimu_10_re, Chimu_30_im);\
    Chi_10_im = _mm512_add_pd(Chimu_10_im, Chimu_30_re);\
    Chi_11_re = _mm512_sub_pd(Chimu_11_re, Chimu_31_im);\
    Chi_11_im = _mm512_add_pd(Chimu_11_im, Chimu_31_re);\
    Chi_12_re = _mm512_sub_pd(Chimu_12_re, Chimu_32_im);\
    Chi_12_im = _mm512_add_pd(Chimu_12_im, Chimu_32_re);

#define TM_PROJ \
    Chi_00_re = _mm512_sub_pd(Chimu_00_re, Chimu_20_re);\
    Chi_00_im = _mm512_sub_pd(Chimu_00_im, Chimu_20_im);\
    Chi_01_re = _mm512_sub_pd(Chimu_01_re, Chimu_21_re);\
    Chi_01_im = _mm512_sub_pd(Chimu_01_im, Chimu_21_im);\
    Chi_02_re = _mm512_sub_pd(Chimu_02_re, Chimu_22_re);\
    Chi_02_im = _mm512_sub_pd(Chimu_02_im, Chimu_22_im);\
    Chi_10_re = _mm512_sub_pd(Chimu_10_re, Chimu_30_re);\
    Chi_10_im = _mm512_sub_pd(Chimu_10_im, Chimu_30_im);\
    Chi_11_re = _mm512_sub_pd(Chimu_11_re, Chimu_31_re);\
    Chi_11_im = _mm512_sub_pd(Chimu_11_im, Chimu_31_im);\
    Chi_12_re = _mm512_sub_pd(Chimu_12_re, Chimu_32_re);\
    Chi_12_im = _mm512_sub_pd(Chimu_12_im, Chimu_32_im);

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
    result_20_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_10_re);\
    result_21_re = UChi_11_im;\
    result_21_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_11_re);\
    result_22_re = UChi_12_im;\
    result_22_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_12_re);\
    result_30_re = UChi_00_im;\
    result_30_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_00_re);\
    result_31_re = UChi_01_im;\
    result_31_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_01_re);\
    result_32_re = UChi_02_im;\
    result_32_im = _mm512_sub_pd(_mm512_setzero_pd(), UChi_02_re);

#define XP_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_add_pd(result_20_re, UChi_10_im);\
    result_20_im = _mm512_sub_pd(result_20_im, UChi_10_re);\
    result_21_re = _mm512_add_pd(result_21_re, UChi_11_im);\
    result_21_im = _mm512_sub_pd(result_21_im, UChi_11_re);\
    result_22_re = _mm512_add_pd(result_22_re, UChi_12_im);\
    result_22_im = _mm512_sub_pd(result_22_im, UChi_12_re);\
    result_30_re = _mm512_add_pd(result_30_re, UChi_00_im);\
    result_30_im = _mm512_sub_pd(result_30_im, UChi_00_re);\
    result_31_re = _mm512_add_pd(result_31_re, UChi_01_im);\
    result_31_im = _mm512_sub_pd(result_31_im, UChi_01_re);\
    result_32_re = _mm512_add_pd(result_32_re, UChi_02_im);\
    result_32_im = _mm512_sub_pd(result_32_im, UChi_02_re);

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
    result_20_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_10_im);\
    result_20_im = UChi_10_re;\
    result_21_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_11_im);\
    result_21_im = UChi_11_re;\
    result_22_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_12_im);\
    result_22_im = UChi_12_re;\
    result_30_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_00_im);\
    result_30_im = UChi_00_re;\
    result_31_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_01_im);\
    result_31_im = UChi_01_re;\
    result_32_re = _mm512_sub_pd(_mm512_setzero_pd(), UChi_02_im);\
    result_32_im = UChi_02_re;

#define XM_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_sub_pd(result_20_re, UChi_10_im);\
    result_20_im = _mm512_add_pd(result_20_im, UChi_10_re);\
    result_21_re = _mm512_sub_pd(result_21_re, UChi_11_im);\
    result_21_im = _mm512_add_pd(result_21_im, UChi_11_re);\
    result_22_re = _mm512_sub_pd(result_22_re, UChi_12_im);\
    result_22_im = _mm512_add_pd(result_22_im, UChi_12_re);\
    result_30_re = _mm512_sub_pd(result_30_re, UChi_00_im);\
    result_30_im = _mm512_add_pd(result_30_im, UChi_00_re);\
    result_31_re = _mm512_sub_pd(result_31_re, UChi_01_im);\
    result_31_im = _mm512_add_pd(result_31_im, UChi_01_re);\
    result_32_re = _mm512_sub_pd(result_32_re, UChi_02_im);\
    result_32_im = _mm512_add_pd(result_32_im, UChi_02_re);

#define YP_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_add_pd(result_20_re, UChi_10_re);\
    result_20_im = _mm512_add_pd(result_20_im, UChi_10_im);\
    result_21_re = _mm512_add_pd(result_21_re, UChi_11_re);\
    result_21_im = _mm512_add_pd(result_21_im, UChi_11_im);\
    result_22_re = _mm512_add_pd(result_22_re, UChi_12_re);\
    result_22_im = _mm512_add_pd(result_22_im, UChi_12_im);\
    result_30_re = _mm512_sub_pd(result_30_re, UChi_00_re);\
    result_30_im = _mm512_sub_pd(result_30_im, UChi_00_im);\
    result_31_re = _mm512_sub_pd(result_31_re, UChi_01_re);\
    result_31_im = _mm512_sub_pd(result_31_im, UChi_01_im);\
    result_32_re = _mm512_sub_pd(result_32_re, UChi_02_re);\
    result_32_im = _mm512_sub_pd(result_32_im, UChi_02_im);

#define YM_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_sub_pd(result_20_re, UChi_10_re);\
    result_20_im = _mm512_sub_pd(result_20_im, UChi_10_im);\
    result_21_re = _mm512_sub_pd(result_21_re, UChi_11_re);\
    result_21_im = _mm512_sub_pd(result_21_im, UChi_11_im);\
    result_22_re = _mm512_sub_pd(result_22_re, UChi_12_re);\
    result_22_im = _mm512_sub_pd(result_22_im, UChi_12_im);\
    result_30_re = _mm512_add_pd(result_30_re, UChi_00_re);\
    result_30_im = _mm512_add_pd(result_30_im, UChi_00_im);\
    result_31_re = _mm512_add_pd(result_31_re, UChi_01_re);\
    result_31_im = _mm512_add_pd(result_31_im, UChi_01_im);\
    result_32_re = _mm512_add_pd(result_32_re, UChi_02_re);\
    result_32_im = _mm512_add_pd(result_32_im, UChi_02_im);

#define ZP_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_add_pd(result_20_re, UChi_00_im);\
    result_20_im = _mm512_sub_pd(result_20_im, UChi_00_re);\
    result_21_re = _mm512_add_pd(result_21_re, UChi_01_im);\
    result_21_im = _mm512_sub_pd(result_21_im, UChi_01_re);\
    result_22_re = _mm512_add_pd(result_22_re, UChi_02_im);\
    result_22_im = _mm512_sub_pd(result_22_im, UChi_02_re);\
    result_30_re = _mm512_sub_pd(result_30_re, UChi_10_im);\
    result_30_im = _mm512_add_pd(result_30_im, UChi_10_re);\
    result_31_re = _mm512_sub_pd(result_31_re, UChi_11_im);\
    result_31_im = _mm512_add_pd(result_31_im, UChi_11_re);\
    result_32_re = _mm512_sub_pd(result_32_re, UChi_12_im);\
    result_32_im = _mm512_add_pd(result_32_im, UChi_12_re);

#define ZM_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_sub_pd(result_20_re, UChi_00_im);\
    result_20_im = _mm512_add_pd(result_20_im, UChi_00_re);\
    result_21_re = _mm512_sub_pd(result_21_re, UChi_01_im);\
    result_21_im = _mm512_add_pd(result_21_im, UChi_01_re);\
    result_22_re = _mm512_sub_pd(result_22_re, UChi_02_im);\
    result_22_im = _mm512_add_pd(result_22_im, UChi_02_re);\
    result_30_re = _mm512_add_pd(result_30_re, UChi_10_im);\
    result_30_im = _mm512_sub_pd(result_30_im, UChi_10_re);\
    result_31_re = _mm512_add_pd(result_31_re, UChi_11_im);\
    result_31_im = _mm512_sub_pd(result_31_im, UChi_11_re);\
    result_32_re = _mm512_add_pd(result_32_re, UChi_12_im);\
    result_32_im = _mm512_sub_pd(result_32_im, UChi_12_re);

#define TP_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_add_pd(result_20_re, UChi_00_re);\
    result_20_im = _mm512_add_pd(result_20_im, UChi_00_im);\
    result_21_re = _mm512_add_pd(result_21_re, UChi_01_re);\
    result_21_im = _mm512_add_pd(result_21_im, UChi_01_im);\
    result_22_re = _mm512_add_pd(result_22_re, UChi_02_re);\
    result_22_im = _mm512_add_pd(result_22_im, UChi_02_im);\
    result_30_re = _mm512_add_pd(result_30_re, UChi_10_re);\
    result_30_im = _mm512_add_pd(result_30_im, UChi_10_im);\
    result_31_re = _mm512_add_pd(result_31_re, UChi_11_re);\
    result_31_im = _mm512_add_pd(result_31_im, UChi_11_im);\
    result_32_re = _mm512_add_pd(result_32_re, UChi_12_re);\
    result_32_im = _mm512_add_pd(result_32_im, UChi_12_im);

#define TM_RECON_ACCUM\
    result_00_re = _mm512_add_pd(result_00_re, UChi_00_re);\
    result_00_im = _mm512_add_pd(result_00_im, UChi_00_im);\
    result_01_re = _mm512_add_pd(result_01_re, UChi_01_re);\
    result_01_im = _mm512_add_pd(result_01_im, UChi_01_im);\
    result_02_re = _mm512_add_pd(result_02_re, UChi_02_re);\
    result_02_im = _mm512_add_pd(result_02_im, UChi_02_im);\
    result_10_re = _mm512_add_pd(result_10_re, UChi_10_re);\
    result_10_im = _mm512_add_pd(result_10_im, UChi_10_im);\
    result_11_re = _mm512_add_pd(result_11_re, UChi_11_re);\
    result_11_im = _mm512_add_pd(result_11_im, UChi_11_im);\
    result_12_re = _mm512_add_pd(result_12_re, UChi_12_re);\
    result_12_im = _mm512_add_pd(result_12_im, UChi_12_im);\
    result_20_re = _mm512_sub_pd(result_20_re, UChi_00_re);\
    result_20_im = _mm512_sub_pd(result_20_im, UChi_00_im);\
    result_21_re = _mm512_sub_pd(result_21_re, UChi_01_re);\
    result_21_im = _mm512_sub_pd(result_21_im, UChi_01_im);\
    result_22_re = _mm512_sub_pd(result_22_re, UChi_02_re);\
    result_22_im = _mm512_sub_pd(result_22_im, UChi_02_im);\
    result_30_re = _mm512_sub_pd(result_30_re, UChi_10_re);\
    result_30_im = _mm512_sub_pd(result_30_im, UChi_10_im);\
    result_31_re = _mm512_sub_pd(result_31_re, UChi_11_re);\
    result_31_im = _mm512_sub_pd(result_31_im, UChi_11_im);\
    result_32_re = _mm512_sub_pd(result_32_re, UChi_12_re);\
    result_32_im = _mm512_sub_pd(result_32_im, UChi_12_im);

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
    _mm512_stream_pd((double*)(base + 64 * 0), result_00_re);\
    _mm512_stream_pd((double*)(base + 64 * 1), result_00_im);\
    _mm512_stream_pd((double*)(base + 64 * 2), result_01_re);\
    _mm512_stream_pd((double*)(base + 64 * 3), result_01_im);\
    _mm512_stream_pd((double*)(base + 64 * 4), result_02_re);\
    _mm512_stream_pd((double*)(base + 64 * 5), result_02_im);\
    _mm512_stream_pd((double*)(base + 64 * 6), result_10_re);\
    _mm512_stream_pd((double*)(base + 64 * 7), result_10_im);\
    _mm512_stream_pd((double*)(base + 64 * 8), result_11_re);\
    _mm512_stream_pd((double*)(base + 64 * 9), result_11_im);\
    _mm512_stream_pd((double*)(base + 64 * 10), result_12_re);\
    _mm512_stream_pd((double*)(base + 64 * 11), result_12_im);\
    _mm512_stream_pd((double*)(base + 64 * 12), result_20_re);\
    _mm512_stream_pd((double*)(base + 64 * 13), result_20_im);\
    _mm512_stream_pd((double*)(base + 64 * 14), result_21_re);\
    _mm512_stream_pd((double*)(base + 64 * 15), result_21_im);\
    _mm512_stream_pd((double*)(base + 64 * 16), result_22_re);\
    _mm512_stream_pd((double*)(base + 64 * 17), result_22_im);\
    _mm512_stream_pd((double*)(base + 64 * 18), result_30_re);\
    _mm512_stream_pd((double*)(base + 64 * 19), result_30_im);\
    _mm512_stream_pd((double*)(base + 64 * 20), result_31_re);\
    _mm512_stream_pd((double*)(base + 64 * 21), result_31_im);\
    _mm512_stream_pd((double*)(base + 64 * 22), result_32_re);\
    _mm512_stream_pd((double*)(base + 64 * 23), result_32_im);\
  }

#define LOAD_T		\
  { const SiteSpinor & ref (in[0]);	base = (uint64_t)ref; \
    DEBUG1_re = _mm512_load_pd((double*)(base + 64 * 0));\
    DEBUG1_im = _mm512_load_pd((double*)(base + 64 * 1));\
    DEBUG2_re = _mm512_load_pd((double*)(base + 64 * 2));\
    DEBUG2_im = _mm512_load_pd((double*)(base + 64 * 3));\
    DEBUG3_re = _mm512_load_pd((double*)(base + 64 * 4));\
    DEBUG3_im = _mm512_load_pd((double*)(base + 64 * 5));\
    DEBUG4_re = _mm512_load_pd((double*)(base + 64 * 6));\
    DEBUG4_im = _mm512_load_pd((double*)(base + 64 * 7));}

#define STORE_T				\
  {	SiteSpinor & ref (out[0]);	base = (uint64_t)ref;		\
    _mm512_stream_pd((double*)(base + 64 * 0), DEBUG1_re);\
    _mm512_stream_pd((double*)(base + 64 * 1), DEBUG1_im);\
    _mm512_stream_pd((double*)(base + 64 * 2), DEBUG2_re);\
    _mm512_stream_pd((double*)(base + 64 * 3), DEBUG2_im);\
    _mm512_stream_pd((double*)(base + 64 * 4), DEBUG3_re);\
    _mm512_stream_pd((double*)(base + 64 * 5), DEBUG3_im);\
    _mm512_stream_pd((double*)(base + 64 * 6), DEBUG4_re);\
    _mm512_stream_pd((double*)(base + 64 * 7), DEBUG4_im);\
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
    _mm512_stream_pd((double*)(base + 64 * 0), DEBUG1_re);\
    _mm512_stream_pd((double*)(base + 64 * 1), DEBUG1_im);\
    _mm512_stream_pd((double*)(base + 64 * 2), DEBUG2_re);\
    _mm512_stream_pd((double*)(base + 64 * 3), DEBUG2_im);\
    _mm512_stream_pd((double*)(base + 64 * 4), DEBUG3_re);\
    _mm512_stream_pd((double*)(base + 64 * 5), DEBUG3_im);\
    _mm512_stream_pd((double*)(base + 64 * 6), DEBUG4_re);\
    _mm512_stream_pd((double*)(base + 64 * 7), DEBUG4_im);\
      std::cout << "XX " << ref[0][0] << std::endl;		\
      std::cout << "XX " << ref[0][1] << std::endl;		\
      std::cout << "XX " << ref[0][2] << std::endl;		\
      std::cout << "XX " << ref[1][0] << std::endl;		\
    }


#define HAND_DECLARATIONS(Simd)			\
    __m512d result_00_re;\
    __m512d result_00_im;\
    __m512d result_01_re;\
    __m512d result_01_im;\
    __m512d result_02_re;\
    __m512d result_02_im;\
    __m512d result_10_re;\
    __m512d result_10_im;\
    __m512d result_11_re;\
    __m512d result_11_im;\
    __m512d result_12_re;\
    __m512d result_12_im;\
    __m512d result_20_re;\
    __m512d result_20_im;\
    __m512d result_21_re;\
    __m512d result_21_im;\
    __m512d result_22_re;\
    __m512d result_22_im;\
    __m512d result_30_re;\
    __m512d result_30_im;\
    __m512d result_31_re;\
    __m512d result_31_im;\
    __m512d result_32_re;\
    __m512d result_32_im;\
    __m512d Chi_00_re;\
    __m512d Chi_00_im;\
    __m512d Chi_01_re;\
    __m512d Chi_01_im;\
    __m512d Chi_02_re;\
    __m512d Chi_02_im;\
    __m512d Chi_10_re;\
    __m512d Chi_10_im;\
    __m512d Chi_11_re;\
    __m512d Chi_11_im;\
    __m512d Chi_12_re;\
    __m512d Chi_12_im;\
    __m512d UChi_00_re;\
    __m512d UChi_00_im;\
    __m512d UChi_01_re;\
    __m512d UChi_01_im;\
    __m512d UChi_02_re;\
    __m512d UChi_02_im;\
    __m512d UChi_10_re;\
    __m512d UChi_10_im;\
    __m512d UChi_11_re;\
    __m512d UChi_11_im;\
    __m512d UChi_12_re;\
    __m512d UChi_12_im;\
    __m512d U_00_re;\
    __m512d U_00_im;\
    __m512d U_10_re;\
    __m512d U_10_im;\
    __m512d U_20_re;\
    __m512d U_20_im;\
    __m512d U_01_re;\
    __m512d U_01_im;\
    __m512d U_11_re;\
    __m512d U_11_im;\
    __m512d U_21_re;\
    __m512d U_21_im;\
    __m512d Chimu_00_re;\
    __m512d Chimu_00_im;\
    __m512d Chimu_01_re;\
    __m512d Chimu_01_im;\
    __m512d Chimu_02_re;\
    __m512d Chimu_02_im;\
    __m512d Chimu_10_re;\
    __m512d Chimu_10_im;\
    __m512d Chimu_11_re;\
    __m512d Chimu_11_im;\
    __m512d Chimu_12_re;\
    __m512d Chimu_12_im;\
    __m512d Chimu_20_re;\
    __m512d Chimu_20_im;\
    __m512d Chimu_21_re;\
    __m512d Chimu_21_im;\
    __m512d Chimu_22_re;\
    __m512d Chimu_22_im;\
    __m512d Chimu_30_re;\
    __m512d Chimu_30_im;\
    __m512d Chimu_31_re;\
    __m512d Chimu_31_im;\
    __m512d Chimu_32_re;\
    __m512d Chimu_32_im;\
    __m512d tmp_re;\
    __m512d tmp_im;\
    __m512d DEBUG1_re;\
    __m512d DEBUG1_im;\
    __m512d DEBUG2_re;\
    __m512d DEBUG2_im;\
    __m512d DEBUG3_re;\
    __m512d DEBUG3_im;\
    __m512d DEBUG4_re;\
    __m512d DEBUG4_im;\




/*
#define Chimu_00_re Chi_00_re
#define Chimu_00_im Chi_00_im
#define Chimu_01_re Chi_01_re
#define Chimu_01_im Chi_01_im
#define Chimu_02_re Chi_02_re
#define Chimu_02_im Chi_02_im
#define Chimu_10_re Chi_10_re
#define Chimu_10_im Chi_10_im
#define Chimu_11_re Chi_11_re
#define Chimu_11_im Chi_11_im
#define Chimu_12_re Chi_12_re
#define Chimu_12_im Chi_12_im
#define Chimu_20_re UChi_00_re
#define Chimu_20_im UChi_00_im
#define Chimu_21_re UChi_01_re
#define Chimu_21_im UChi_01_im
#define Chimu_22_re UChi_02_re
#define Chimu_22_im UChi_02_im
#define Chimu_30_re UChi_10_re
#define Chimu_30_im UChi_10_im
#define Chimu_31_re UChi_11_re
#define Chimu_31_im UChi_11_im
#define Chimu_32_re UChi_12_re
#define Chimu_32_im UChi_12_im
*/

#ifndef GRID_SYCL
#define GRID_OMP_THREAD
#endif

#ifdef GRID_OMP_THREAD
template<class SimdVec>
double dslash_kernel_cpu(int nrep,SimdVec *Up,SimdVec *outp,SimdVec *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm,int psi_pf_dist_L1, int psi_pf_dist_L2, int u_pf_dist_L2)
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
#ifdef XXOMP
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
    DEBUG1_re = _mm512_mul_pd(DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = _mm512_mul_pd(DEBUG2_re, DEBUG3_im);\
    DEBUG1_re = _mm512_fnmadd_pd(DEBUG2_im, DEBUG3_im, DEBUG1_re);\
    DEBUG1_im = _mm512_fmadd_pd(DEBUG2_im, DEBUG3_re, DEBUG1_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += DEBUG2*DEBUG3" << std::endl;
    DEBUG1_re = _mm512_fmadd_pd(DEBUG2_re, DEBUG3_re, DEBUG1_re);\
    DEBUG1_im = _mm512_fmadd_pd(DEBUG2_re, DEBUG3_im, DEBUG1_im);\
    DEBUG1_re = _mm512_fnmadd_pd(DEBUG2_im, DEBUG3_im, DEBUG1_re);\
    DEBUG1_im = _mm512_fmadd_pd(DEBUG2_im, DEBUG3_re, DEBUG1_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2-DEBUG3" << std::endl;
    DEBUG1_re = _mm512_sub_pd(DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = _mm512_sub_pd(DEBUG2_im, DEBUG3_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 -= DEBUG2" << std::endl;
    DEBUG1_re = _mm512_sub_pd(DEBUG1_re, DEBUG2_re);\
    DEBUG1_im = _mm512_sub_pd(DEBUG1_im, DEBUG2_im);
    STOREPRINT_T;
        
    std::cout << "XX DEBUG1 = DEBUG2+DEBUG3" << std::endl;
    DEBUG1_re = _mm512_add_pd(DEBUG2_re, DEBUG3_re);\
    DEBUG1_im = _mm512_add_pd(DEBUG2_im, DEBUG3_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += DEBUG2" << std::endl;
    DEBUG1_re = _mm512_add_pd(DEBUG1_re, DEBUG2_re);\
    DEBUG1_im = _mm512_add_pd(DEBUG1_im, DEBUG2_im);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = timesI(DEBUG2)" << std::endl;
    DEBUG1_re = _mm512_sub_pd(_mm512_setzero_pd(), DEBUG2_im);\
    DEBUG1_im = DEBUG2_re;
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2+timesI(DEBUG3)" << std::endl;
    DEBUG1_re = _mm512_sub_pd(DEBUG2_re, DEBUG3_im);\
    DEBUG1_im = _mm512_add_pd(DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 = DEBUG2-timesI(DEBUG3)" << std::endl;
    DEBUG1_re = _mm512_add_pd(DEBUG2_re, DEBUG3_im);\
    DEBUG1_im = _mm512_sub_pd(DEBUG2_im, DEBUG3_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 += timesI(DEBUG2)" << std::endl;
    DEBUG1_re = _mm512_sub_pd(DEBUG1_re, DEBUG2_im);\
    DEBUG1_im = _mm512_add_pd(DEBUG1_im, DEBUG2_re);
    STOREPRINT_T;

    std::cout << "XX DEBUG1 -= timesI(DEBUG2)" << std::endl;
    DEBUG1_re = _mm512_add_pd(DEBUG1_re, DEBUG2_im);\
    DEBUG1_im = _mm512_sub_pd(DEBUG1_im, DEBUG2_re);
    STOREPRINT_T;

    std::cout << "XX " << std::endl;
    DEBUG1_re = DEBUG2_im;\
    DEBUG1_im = _mm512_sub_pd(_mm512_setzero_pd(), DEBUG2_re);
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
