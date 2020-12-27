/*
 * SVETemplate5.h
 *
 * - introduced permutes
 * (reverted complex multiplication split into 2 rounds due to performance penalty (single thread 206.303 cy instead of 193.879))

$ for i in `seq 1 12` ; do OMP_NUM_THREADS=$i ./bench.rrii.sve.intrinsics.gcc 32 100 2> /dev/null | grep XX1 ; done
1  32  16x16x16x32x8  12.255  21.2761  193.879  1551.04  XX1
2  32  16x16x16x32x8  21.9419  19.0468  216.572  1732.57  XX1
3  32  16x16x16x32x8  31.819  18.4138  224.017  1792.14  XX1
4  32  16x16x16x32x8  40.9357  17.7672  232.169  1857.35  XX1
5  32  16x16x16x32x8  49.5587  17.2079  239.716  1917.73  XX1
6  32  16x16x16x32x8  58.0701  16.8027  245.496  1963.97  XX1
7  32  16x16x16x32x8  65.9289  16.3514  252.272  2018.18  XX1
8  32  16x16x16x32x8  73.6632  15.9859  258.039  2064.32  XX1
9  32  16x16x16x32x8  80.8846  15.6027  264.377  2115.01  XX1
10  32  16x16x16x32x8  87.57  15.2031  271.326  2170.61  XX1
11  32  16x16x16x32x8  93.8792  14.8168  278.4  2227.2  XX1
12  32  16x16x16x32x8  99.7469  14.431  285.844  2286.75  XX1

 * No significant impact of permute on performance
 */

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
    Chimu_32=coalescedReadPermute<ptype>(ref[3][2],perm,mylane); }

#define PERMUTE_DIR(dir) ;

#else
#define LOAD_CHIMU(ptype)		\
  { const SiteSpinor & ref (in[offset]);	base = (uint64_t)ref; \
    Chimu_00=coalescedRead(ref[0][0],mylane);	\
    Chimu_01=coalescedRead(ref[0][1],mylane);	\
    Chimu_02=coalescedRead(ref[0][2],mylane);	\
    Chimu_10=coalescedRead(ref[1][0],mylane);	\
    Chimu_11=coalescedRead(ref[1][1],mylane);	\
    Chimu_12=coalescedRead(ref[1][2],mylane);	\
    Chimu_20=coalescedRead(ref[2][0],mylane);	\
    Chimu_21=coalescedRead(ref[2][1],mylane);	\
    Chimu_22=coalescedRead(ref[2][2],mylane);	\
    Chimu_30=coalescedRead(ref[3][0],mylane);	\
    Chimu_31=coalescedRead(ref[3][1],mylane);	\
    Chimu_32=coalescedRead(ref[3][2],mylane); }

/*
#define PERMUTE_DIR(dir)			\
      permute##dir(Chi_00,Chi_00);\
      permute##dir(Chi_01,Chi_01);\
      permute##dir(Chi_02,Chi_02);\
      permute##dir(Chi_10,Chi_10);\
      permute##dir(Chi_11,Chi_11);\
      permute##dir(Chi_12,Chi_12);
*/

#define PERMUTE_DIR(dir) \
  if      (dir == 0) table = svld1(pg1, (uint64_t*)&lut[0]); \
  else if (dir == 1) table = svld1(pg1, (uint64_t*)&lut[1]); \
  else if (dir == 2) table = svld1(pg1, (uint64_t*)&lut[2]); \
  else if (dir == 3) table = svld1(pg1, (uint64_t*)&lut[3]); \
  Chi_00 = svtbl(Chi_00, table);    \
  Chi_01 = svtbl(Chi_01, table);    \
  Chi_02 = svtbl(Chi_02, table);    \
  Chi_10 = svtbl(Chi_10, table);    \
  Chi_11 = svtbl(Chi_11, table);    \
  Chi_12 = svtbl(Chi_12, table);

#endif


#define MULT_2SPIN(A)\
  { auto & ref(U[sU][A]); base = (uint64_t)ref;	\
    U_00=coalescedRead(ref[0][0],mylane);				\
    U_10=coalescedRead(ref[1][0],mylane);				\
    U_20=coalescedRead(ref[2][0],mylane);				\
    U_01=coalescedRead(ref[0][1],mylane);				\
    U_11=coalescedRead(ref[1][1],mylane);				\
    U_21=coalescedRead(ref[2][1],mylane);				\
    UChi_00 = U_00*Chi_00;                                      \
    UChi_10 = U_00*Chi_10;                                      \
    UChi_01 = U_10*Chi_00;                                      \
    UChi_11 = U_10*Chi_10;                                      \
    UChi_02 = U_20*Chi_00;                                      \
    UChi_12 = U_20*Chi_10;                                      \
    UChi_00+= U_01*Chi_01;                                      \
    UChi_10+= U_01*Chi_11;                                      \
    UChi_01+= U_11*Chi_01;                                      \
    UChi_11+= U_11*Chi_11;                                      \
    UChi_02+= U_21*Chi_01;                                      \
    UChi_12+= U_21*Chi_11;                                      \
    U_00=coalescedRead(ref[0][2],mylane);				\
    U_10=coalescedRead(ref[1][2],mylane);				\
    U_20=coalescedRead(ref[2][2],mylane);				\
    UChi_00+= U_00*Chi_02;                                      \
    UChi_10+= U_00*Chi_12;                                      \
    UChi_01+= U_10*Chi_02;                                      \
    UChi_11+= U_10*Chi_12;                                      \
    UChi_02+= U_20*Chi_02;                                      \
    UChi_12+= U_20*Chi_12;}

//      hspin(0)=fspin(0)+timesI(fspin(3));
//      hspin(1)=fspin(1)+timesI(fspin(2));
#define XP_PROJ \
    Chi_00 = Chimu_00+timesI(Chimu_30);\
    Chi_01 = Chimu_01+timesI(Chimu_31);\
    Chi_02 = Chimu_02+timesI(Chimu_32);\
    Chi_10 = Chimu_10+timesI(Chimu_20);\
    Chi_11 = Chimu_11+timesI(Chimu_21);\
    Chi_12 = Chimu_12+timesI(Chimu_22);

#define YP_PROJ \
    Chi_00 = Chimu_00-Chimu_30;\
    Chi_01 = Chimu_01-Chimu_31;\
    Chi_02 = Chimu_02-Chimu_32;\
    Chi_10 = Chimu_10+Chimu_20;\
    Chi_11 = Chimu_11+Chimu_21;\
    Chi_12 = Chimu_12+Chimu_22;

#define ZP_PROJ \
  Chi_00 = Chimu_00+timesI(Chimu_20);		\
  Chi_01 = Chimu_01+timesI(Chimu_21);		\
  Chi_02 = Chimu_02+timesI(Chimu_22);		\
  Chi_10 = Chimu_10-timesI(Chimu_30);		\
  Chi_11 = Chimu_11-timesI(Chimu_31);		\
  Chi_12 = Chimu_12-timesI(Chimu_32);

#define TP_PROJ \
  Chi_00 = Chimu_00+Chimu_20;		\
  Chi_01 = Chimu_01+Chimu_21;		\
  Chi_02 = Chimu_02+Chimu_22;		\
  Chi_10 = Chimu_10+Chimu_30;		\
  Chi_11 = Chimu_11+Chimu_31;		\
  Chi_12 = Chimu_12+Chimu_32;


//      hspin(0)=fspin(0)-timesI(fspin(3));
//      hspin(1)=fspin(1)-timesI(fspin(2));
#define XM_PROJ \
    Chi_00 = Chimu_00-timesI(Chimu_30);\
    Chi_01 = Chimu_01-timesI(Chimu_31);\
    Chi_02 = Chimu_02-timesI(Chimu_32);\
    Chi_10 = Chimu_10-timesI(Chimu_20);\
    Chi_11 = Chimu_11-timesI(Chimu_21);\
    Chi_12 = Chimu_12-timesI(Chimu_22);

#define YM_PROJ \
    Chi_00 = Chimu_00+Chimu_30;\
    Chi_01 = Chimu_01+Chimu_31;\
    Chi_02 = Chimu_02+Chimu_32;\
    Chi_10 = Chimu_10-Chimu_20;\
    Chi_11 = Chimu_11-Chimu_21;\
    Chi_12 = Chimu_12-Chimu_22;

#define ZM_PROJ \
  Chi_00 = Chimu_00-timesI(Chimu_20);		\
  Chi_01 = Chimu_01-timesI(Chimu_21);		\
  Chi_02 = Chimu_02-timesI(Chimu_22);		\
  Chi_10 = Chimu_10+timesI(Chimu_30);		\
  Chi_11 = Chimu_11+timesI(Chimu_31);		\
  Chi_12 = Chimu_12+timesI(Chimu_32);

#define TM_PROJ \
  Chi_00 = Chimu_00-Chimu_20;		\
  Chi_01 = Chimu_01-Chimu_21;		\
  Chi_02 = Chimu_02-Chimu_22;		\
  Chi_10 = Chimu_10-Chimu_30;		\
  Chi_11 = Chimu_11-Chimu_31;		\
  Chi_12 = Chimu_12-Chimu_32;

//      fspin(0)=hspin(0);
//      fspin(1)=hspin(1);
//      fspin(2)=timesMinusI(hspin(1));
//      fspin(3)=timesMinusI(hspin(0));

#define XP_RECON\
  result_00 = UChi_00;\
  result_01 = UChi_01;\
  result_02 = UChi_02;\
  result_10 = UChi_10;\
  result_11 = UChi_11;\
  result_12 = UChi_12;\
  result_20 = timesMinusI(UChi_10);\
  result_21 = timesMinusI(UChi_11);\
  result_22 = timesMinusI(UChi_12);\
  result_30 = timesMinusI(UChi_00);\
  result_31 = timesMinusI(UChi_01);\
  result_32 = timesMinusI(UChi_02);

#define XP_RECON_ACCUM\
  result_00+=UChi_00;\
  result_01+=UChi_01;\
  result_02+=UChi_02;\
  result_10+=UChi_10;\
  result_11+=UChi_11;\
  result_12+=UChi_12;\
  result_20-=timesI(UChi_10);\
  result_21-=timesI(UChi_11);\
  result_22-=timesI(UChi_12);\
  result_30-=timesI(UChi_00);\
  result_31-=timesI(UChi_01);\
  result_32-=timesI(UChi_02);

#define XM_RECON\
  result_00 = UChi_00;\
  result_01 = UChi_01;\
  result_02 = UChi_02;\
  result_10 = UChi_10;\
  result_11 = UChi_11;\
  result_12 = UChi_12;\
  result_20 = timesI(UChi_10);\
  result_21 = timesI(UChi_11);\
  result_22 = timesI(UChi_12);\
  result_30 = timesI(UChi_00);\
  result_31 = timesI(UChi_01);\
  result_32 = timesI(UChi_02);

#define XM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= timesI(UChi_10);\
  result_21+= timesI(UChi_11);\
  result_22+= timesI(UChi_12);\
  result_30+= timesI(UChi_00);\
  result_31+= timesI(UChi_01);\
  result_32+= timesI(UChi_02);

#define YP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= UChi_10;\
  result_21+= UChi_11;\
  result_22+= UChi_12;\
  result_30-= UChi_00;\
  result_31-= UChi_01;\
  result_32-= UChi_02;

#define YM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= UChi_10;\
  result_21-= UChi_11;\
  result_22-= UChi_12;\
  result_30+= UChi_00;\
  result_31+= UChi_01;\
  result_32+= UChi_02;

#define ZP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= timesI(UChi_00);			\
  result_21-= timesI(UChi_01);			\
  result_22-= timesI(UChi_02);			\
  result_30+= timesI(UChi_10);			\
  result_31+= timesI(UChi_11);			\
  result_32+= timesI(UChi_12);

#define ZM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= timesI(UChi_00);			\
  result_21+= timesI(UChi_01);			\
  result_22+= timesI(UChi_02);			\
  result_30-= timesI(UChi_10);			\
  result_31-= timesI(UChi_11);			\
  result_32-= timesI(UChi_12);

#define TP_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20+= UChi_00;			\
  result_21+= UChi_01;			\
  result_22+= UChi_02;			\
  result_30+= UChi_10;			\
  result_31+= UChi_11;			\
  result_32+= UChi_12;

#define TM_RECON_ACCUM\
  result_00+= UChi_00;\
  result_01+= UChi_01;\
  result_02+= UChi_02;\
  result_10+= UChi_10;\
  result_11+= UChi_11;\
  result_12+= UChi_12;\
  result_20-= UChi_00;	\
  result_21-= UChi_01;	\
  result_22-= UChi_02;	\
  result_30-= UChi_10;	\
  result_31-= UChi_11;	\
  result_32-= UChi_12;

#define HAND_STENCIL_LEG(PROJ,PERM,DIR,RECON)		\
  offset = nbr[ss*8+DIR];				\
  pf_L1  = nbr[ss*8+DIR+1];				\
  pf_L2  = nbr[ssn*8+DIR-1];				\
  perm   = prm[ss*8+DIR];				\
  LOAD_CHIMU(PERM);					\
  PROJ;							\
  if (perm) {						\
    PERMUTE_DIR(PERM);					\
  }							\
  PREFETCH_CHIMU_L2; 					\
  PREFETCH_CHIMU_L1;        \
  MULT_2SPIN(DIR);					\
  if (s == 0) {                                           \
   if ((DIR == 0) || (DIR == 2) || (DIR == 4) || (DIR == 6)) { PREFETCH_GAUGE_L2(DIR); } \
  }                                                       \
  RECON;

#define HAND_RESULT(ss)				\
  {	SiteSpinor & ref (out[ss]);	base = (uint64_t)ref;		\
    coalescedWrite(ref[0][0],result_00,mylane);		\
    coalescedWrite(ref[0][1],result_01,mylane);		\
    coalescedWrite(ref[0][2],result_02,mylane);		\
    coalescedWrite(ref[1][0],result_10,mylane);		\
    coalescedWrite(ref[1][1],result_11,mylane);		\
    coalescedWrite(ref[1][2],result_12,mylane);		\
    coalescedWrite(ref[2][0],result_20,mylane);		\
    coalescedWrite(ref[2][1],result_21,mylane);		\
    coalescedWrite(ref[2][2],result_22,mylane);		\
    coalescedWrite(ref[3][0],result_30,mylane);		\
    coalescedWrite(ref[3][1],result_31,mylane);		\
    coalescedWrite(ref[3][2],result_32,mylane);		\
  }

#define PREFETCH_CHIMU_L2  \
{ const SiteSpinor & ref (in[pf_L2]);	base = (uint64_t)ref; \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(0), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(4), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(8), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(12), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(16), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(20), SV_PLDL2STRM); \
}

#define PREFETCH_CHIMU_L1  \
{ const SiteSpinor & ref (in[pf_L1]);	base = (uint64_t)ref;   \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(0), SV_PLDL1STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(4), SV_PLDL1STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(8), SV_PLDL1STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(12), SV_PLDL1STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(16), SV_PLDL1STRM); \
  svprfd_vnum(pg1, (long*)(base), (int64_t)(20), SV_PLDL1STRM); \
}

// PREFETCH_GAUGE_L2 (prefetch to L2)
#define PREFETCH_GAUGE_L2(A)  \
{ \
  const auto & ref(U[sUn][A]); baseU = (uint64_t)&ref; \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(0), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(4), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(8), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(12), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(16), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(20), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(24), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(28), SV_PLDL2STRM); \
  svprfd_vnum(pg1, (long*)(baseU), (int64_t)(32), SV_PLDL2STRM); \
}

#define HAND_DECLARATIONS(Simd)			\
  Simd result_00;				\
  Simd result_01;				\
  Simd result_02;				\
  Simd result_10;				\
  Simd result_11;				\
  Simd result_12;				\
  Simd result_20;				\
  Simd result_21;				\
  Simd result_22;				\
  Simd result_30;				\
  Simd result_31;				\
  Simd result_32;				\
  Simd Chi_00;					\
  Simd Chi_01;					\
  Simd Chi_02;					\
  Simd Chi_10;					\
  Simd Chi_11;					\
  Simd Chi_12;					\
  Simd UChi_00;					\
  Simd UChi_01;					\
  Simd UChi_02;					\
  Simd UChi_10;					\
  Simd UChi_11;					\
  Simd UChi_12;					\
  Simd U_00;					\
  Simd U_10;					\
  Simd U_20;					\
  Simd U_01;					\
  Simd U_11;					\
  Simd U_21;          \
  Simd Chimu_00;      \
  Simd Chimu_01;      \
  Simd Chimu_02;      \
  Simd Chimu_10;      \
  Simd Chimu_11;      \
  Simd Chimu_12;      \
  Simd Chimu_20;      \
  Simd Chimu_21;      \
  Simd Chimu_22;      \
  Simd Chimu_30;      \
  Simd Chimu_31;      \
  Simd Chimu_32;      \
  const uint64_t lut[4][8] =    \
  { {4,5,6,7,0,1,2,3}, {2,3,0,1,6,7,4,5}, {1,0,3,2,5,4,7,6}, {0,1,2,3,4,5,6,7} }; \
  svuint64_t table; \
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
  for(uint64_t ssite=0;ssite<nsite;ssite++){


    //HAND_DECLARATIONS(svfloat64_t);
    HAND_DECLARATIONS(Simd);
    int mylane=0;
    int offset,perm;
    uint64_t base;
    uint64_t sU = ssite;
    uint64_t sUn = ssite+1;
    if (sUn == nsite) sUn = 0;
    uint64_t ss = sU*Ls;
    uint64_t ssn = ss + 1; // for prefetching to L2
    if (ssn == nsite) ssn = 0;
    uint64_t pf_L1, pf_L2; // pf addresses psi
    uint64_t baseU;        // pf U
    for(uint64_t s=0;s<Ls;s++){
      HAND_STENCIL_LEG(XM_PROJ,3,Xp,XM_RECON);
      HAND_STENCIL_LEG(YM_PROJ,2,Yp,YM_RECON_ACCUM);
      HAND_STENCIL_LEG(ZM_PROJ,1,Zp,ZM_RECON_ACCUM);
      HAND_STENCIL_LEG(TM_PROJ,0,Tp,TM_RECON_ACCUM);
      HAND_STENCIL_LEG(XP_PROJ,3,Xm,XP_RECON_ACCUM);
      HAND_STENCIL_LEG(YP_PROJ,2,Ym,YP_RECON_ACCUM);
      HAND_STENCIL_LEG(ZP_PROJ,1,Zm,ZP_RECON_ACCUM);
      HAND_STENCIL_LEG(TP_PROJ,0,Tm,TP_RECON_ACCUM);
      HAND_RESULT(ss);
      ss++;
      }
    }
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
  uvoid* nbrsvm=(uint64_t *) malloc_shared(_nbrmax*sizeof(uint64_t),q);
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
