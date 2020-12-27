#pragma once
#include <stdio.h>
#include <bitset>

#ifdef AV
#pragma message ("AV")
#include "w.h"
#endif

#ifdef INTRIN
//#pragma message ("INTRIN")
#include <arm_sve.h>
#include "wi.h"
#endif

#ifdef ASM
#pragma message ("ASM")
#include "wa.h"
#endif

/*
static constexpr int Xp = 0;
static constexpr int Yp = 1;
static constexpr int Zp = 2;
static constexpr int Tp = 3;
static constexpr int Xm = 4;
static constexpr int Ym = 5;
static constexpr int Zm = 6;
static constexpr int Tm = 7;
*/
#ifdef KERNEL_DAG
#define DIR0_PROJ    XP_PROJ
#define DIR1_PROJ    YP_PROJ
#define DIR2_PROJ    ZP_PROJ
#define DIR3_PROJ    TP_PROJ
#define DIR4_PROJ    XM_PROJ
#define DIR5_PROJ    YM_PROJ
#define DIR6_PROJ    ZM_PROJ
#define DIR7_PROJ    TM_PROJ
#define DIR0_RECON   XP_RECON
#define DIR1_RECON   YP_RECON_ACCUM
#define DIR2_RECON   ZP_RECON_ACCUM
#define DIR3_RECON   TP_RECON_ACCUM
#define DIR4_RECON   XM_RECON_ACCUM
#define DIR5_RECON   YM_RECON_ACCUM
#define DIR6_RECON   ZM_RECON_ACCUM
#define DIR7_RECON   TM_RECON_ACCUM
#else
#define DIR0_PROJ    XM_PROJ
#define DIR1_PROJ    YM_PROJ
#define DIR2_PROJ    ZM_PROJ
#define DIR3_PROJ    TM_PROJ
#define DIR4_PROJ    XP_PROJ
#define DIR5_PROJ    YP_PROJ
#define DIR6_PROJ    ZP_PROJ
#define DIR7_PROJ    TP_PROJ
#define DIR0_RECON   XM_RECON
#define DIR1_RECON   YM_RECON_ACCUM
#define DIR2_RECON   ZM_RECON_ACCUM
#define DIR3_RECON   TM_RECON_ACCUM
#define DIR4_RECON   XP_RECON_ACCUM
#define DIR5_RECON   YP_RECON_ACCUM
#define DIR6_RECON   ZP_RECON_ACCUM
#define DIR7_RECON   TP_RECON_ACCUM
#endif

#define INTERIOR_AND_EXTERIOR
//#define INTERIOR

////////////////////////////////////////////////////////////////////////////////
// Comms then compute kernel
////////////////////////////////////////////////////////////////////////////////
#ifdef INTERIOR_AND_EXTERIOR

#define ASM_LEG(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON)			\
        offset = nbr[ssn*8+Dir]; {auto & ref(in[offset]); basep = (uint64_t)&ref - 1 * 3*4*64;} \
        offset = nbr[ss*8+NxtDir+1]; {auto & ref(in[offset]); base2 = (uint64_t)&ref;} \
    LOAD_CHIMU(base);                                       \
    LOAD_TABLE(PERMUTE_DIR);                                \
    PROJ;							                        \
  /*   PREFETCH_CHIMU_L1(base); 1245 1359       */                         \
    MAYBEPERM(PERMUTE_DIR,perm);					        \
    /*  offset = nbr[ss*8+NxtDir]; {auto & ref(in[offset]); base = (uint64_t)&ref;} perm = prm[ss*8+NxtDir]; */ \
        perm = prm[ss*8+NxtDir]; \
        base = base2; \
   /* LOAD_GAUGE(Dir); */                                        \
    MULT_2SPIN_1(Dir);					                    \
   /* PREFETCH_CHIMU_L1(base); */                                \
    PREFETCH_CHIMU_L1(base2); /* 1232 1191  1260 990          */                           \
    PREFETCH_CHIMU_L2(basep);                               \
    /* PREFETCH_GAUGE_L1(NxtDir); */                        \
    MULT_2SPIN_2;					                        \
  /*  if (s == 0) {                                  */         \
  /*    if ((Dir == 0) || (Dir == 4)) { PREFETCH_GAUGE_L2(Dir); } */  \
  /*  }                                              */      \
    RECON;

/*
 * FX700, 1 thread
 *
 *           gcc 10.1    armclang 20.3
 *
 *   CONSECUTIVE LOAD_CHIMU
 *
 * No prefetching:
 *   asm     1234
 *   acle    1400
 *
 * w/ PREFETCH_CHIMU(base);
 *   asm     1200
 *   acle    1300
 *
 * PREFETCH_CHIMU_L2(basep);
 *   asm     1240
 *   acle    1420
 *
 * w/ PREFETCH_CHIMU(base);
 * w/ PREFETCH_CHIMU_L2(basep);
 *   asm     1230
 *   acle    1200
 *
 * w/ PREFETCH_CHIMU(base);
 * w/ PREFETCH_CHIMU_L2(basep);
 * w/ PREFETCH_GAUGE_L2(Dir);
 *   asm     1200        1226
 *   acle    1290        1170
 *
 *   INTERLEAVE LOAD_CHIMU
 *
 * w/ PREFETCH_CHIMU(base);
 * w/ PREFETCH_CHIMU_L2(basep);
 *   asm     1215        1245
 *   acle    1200        1110
 */


#define ASM_LEG_XP(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON)		\
    offset = nbr[ss*8+Dir]; {auto & ref(in[offset]); base = (uint64_t)&ref;} perm = prm[ss*8+Dir]; \
  /* PREFETCH_GAUGE_L2(NxtDir); penalty also here */                                  \
  /* PREFETCH1_CHIMU(base); */						            \
  ASM_LEG(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON);

#define RESULT(base,basep) SAVE_RESULT(base,basep);

#endif

////////////////////////////////////////////////////////////////////////////////
// Pre comms kernel -- prefetch like normal because it is mostly right
////////////////////////////////////////////////////////////////////////////////
#ifdef INTERIOR

#define ASM_LEG(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON)			\
      /* basep = st.GetPFInfo(nent,plocal); nent++; */			\
      perm   = prm[ss*8+Dir]; \
      offset = nbr[ssn*8+Dir];\
      {auto & ref(in[offset]); \
      basep = (uint64_t)&ref;}  \
      /* if ( local ) {	*/						\
	LOAD64(%r10,isigns);						\
	PROJ(base);							\
	MAYBEPERM(PERMUTE_DIR,perm);					\
 /* } else if ( st.same_node[Dir] ) {LOAD_CHI(base);}*/			\
      offset = nbr[ss*8+NxtDir];\
       {auto & ref(in[offset]); \
       base = (uint64_t)&ref;}  \
     /* if ( local || st.same_node[Dir] ) */ {				\
	MULT_2SPIN_DIR_PF(Dir,basep);					\
    PREFETCH_CHIMU(base);						\
	LOAD64(%r10,isigns);						\
	RECON;								\
}
/*      }									\
      offset = nbr[ss*8+NxtDir];\
      {auto & ref(in[offset]); \
      base = (uint64_t)&ref;}  \
      PREFETCH_CHIMU(base);						\ */

#define ASM_LEG_XP(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON)			\
  /* base = st.GetInfo(ptype,local,perm,Dir,ent,plocal); ent++;	*/	\
  offset = nbr[ss*8+Dir];\
  {auto & ref(in[offset]); \
  base = (uint64_t)&ref;}  \
  perm   = prm[ss*8+Dir]; \
  /* std::cout << "base: " << base << "  ss: " << ss << "  site: " << ssite << std::endl; */ \
  /* base = (uint64_t)&in[0]; */                  \
  PF_GAUGE(Xp);								\
  PREFETCH1_CHIMU(base);						\
  { ZERO_PSI; }								\
  ASM_LEG(Dir,NxtDir,PERMUTE_DIR,PROJ,RECON)

#define RESULT(base,basep) SAVE_RESULT(base,basep);

#endif

template<class SimdVec>
double dslash_kernel_cpu(int nrep,SimdVec *Up,SimdVec *outp,SimdVec *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm)
{

    #ifdef __ARM_FEATURE_SVE
    //#pragma message("Using Aarch64 clock cycle counter")
      unsigned long long ts, te, freq, cycles;
      asm volatile ("isb; mrs %0, cntfrq_el0" : "=r" (freq));
    #else
      //typedef  std::chrono::system_clock          Clock;
      typedef  std::chrono::high_resolution_clock          Clock;
      typedef  std::chrono::time_point<Clock> TimePoint;
      typedef  std::chrono::microseconds          Usecs;
      TimePoint start;
    #endif
  //std::bitset<8> directions = 0;

  // typedef  std::chrono::system_clock          Clock;
  // typedef  std::chrono::time_point<Clock> TimePoint;
  // typedef  std::chrono::microseconds          Usecs;

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

  //TimePoint start;


  double usec;
  for(int rep=0;rep<nrep+1;rep++){
    if ( rep==1 )
    #ifdef __ARM_FEATURE_SVE
      asm volatile ("isb; mrs %0, cntvct_el0" : "=r" (ts));
    #else
        start = Clock::now();
    #endif
    //start = Clock::now();

//  int ssU = 0;
//  int ss  = 0;

#ifdef OMP
#pragma omp parallel for schedule(static)
#endif
  for(uint64_t ssite=0;ssite<nsite;ssite++){
      int nmu;
      int local,perm, ptype,offset;

      int ss;
      int ssU = 0;
      ssU = ssite;

      uint64_t base;
      uint64_t base2;
      uint64_t basep;
      const uint64_t plocal =(uint64_t) &in[0];

      //COMPLEX_SIGNS(isigns);
      MASK_REGS;
      //int nmax=U.oSites();
      int nmax=nsite;
      //    int sU =lo.Reorder(ssU);
      int sU =ssU;
      int ssn=ssU+1;     if(ssn>=nmax) ssn=0;
      //    int sUn=lo.Reorder(ssn);
      int sUn=ssn;

        for(int s=0;s<Ls;s++) {
          ss =sU*Ls+s;
          ssn=sUn*Ls+s;
          int  ent=ss*8;// 2*Ndim
          int nent=ssn*8;
          // std::cout << "ssite = " << ssite \
          //   << ", sU = " << sU << ", ss = " << ss \
          //   << ", sU_next = " << sUn << ", ss_next = " << ssn \
          //   << ", ent = " << ent << ", nent = " << nent << std::endl;

       ASM_LEG_XP(Xp,Yp,PERMUTE_DIR3,DIR0_PROJ,DIR0_RECON);
          ASM_LEG(Yp,Zp,PERMUTE_DIR2,DIR1_PROJ,DIR1_RECON);
          ASM_LEG(Zp,Tp,PERMUTE_DIR1,DIR2_PROJ,DIR2_RECON);
          ASM_LEG(Tp,Xm,PERMUTE_DIR0,DIR3_PROJ,DIR3_RECON);

          ASM_LEG(Xm,Ym,PERMUTE_DIR3,DIR4_PROJ,DIR4_RECON);
          ASM_LEG(Ym,Zm,PERMUTE_DIR2,DIR5_PROJ,DIR5_RECON);
          ASM_LEG(Zm,Tm,PERMUTE_DIR1,DIR6_PROJ,DIR6_RECON);
          ASM_LEG(Tm,Xp,PERMUTE_DIR0,DIR7_PROJ,DIR7_RECON);

          base = (uint64_t)&out[ss];
          basep = (uint64_t)&out[ssn];

          RESULT(base,basep);


          //if (in_cache == 0)
            // ss++;
        }
        auto in_cache = 0;
        if (in_cache == 0)
            ssU++;

      }
  //}
  }
  #ifdef __ARM_FEATURE_SVE
    asm volatile ("isb; mrs %0, cntvct_el0" : "=r" (te));
    cycles = (unsigned long long)(te - ts);
    usec = 1000000.0 * (double)cycles  / (double)freq;
  #else
    Usecs elapsed = std::chrono::duration_cast<Usecs>(Clock::now()-start);
    usec = elapsed.count();
  #endif

  return usec;
}
