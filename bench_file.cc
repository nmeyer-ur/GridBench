#define  DOUBLE
#if defined(SVE) && defined(RIRI)
#define DATA_SIMD 4  // Size in riri static data
#define EXPAND_SIMD 4 // Target size riri
#endif
#if defined(SVE) && defined(RRII)
#define DATA_SIMD 8  // Size in riri static data
#define EXPAND_SIMD 8  // Target size rrii
#endif

//#if defined(AVX512) //&& defined(RIRI)
#if defined(AVX512) && defined(RIRI)
#define DATA_SIMD 4  // Size of riri data
#define EXPAND_SIMD 4 // Target size riri
#endif
#if defined(AVX512) && defined(RRII)
#define DATA_SIMD 8  // Size of rrii data
#define EXPAND_SIMD 8  // Target size rrii
#endif

/* SVE & AVX512
 *
 * SVE: riri data generation: build Grid with --enable-simd=A64FX;
 * run TableGenerateD;
 * use riri dslash kernel with settings
 *   DATA_SIMD   4
 *   EXPAND_SIMD 4
 *
 * SVE: riri data generation: build Grid with --enable-simd=A64FX;
 * run TableGenerateF;
 * use rrii dslash kernel with settings
 *   DATA_SIMD   8
 *   EXPAND_SIMD 8
 */

// Invoke dslash.s - test for compiler-gsnerated code
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <strings.h>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <cassert>

#include "Simd.h"
#include "WilsonKernelsHand.h"

#ifdef OMP
#include <omp.h>
// some OMP function calls don't work on my laptop, use alternative
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
#endif

#ifdef __x86_64__
#define __SSC_MARK(A) __asm__ __volatile__ ("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 " ::"i"(A):"%ebx")
#else
#define __SSC_MARK(A)
#endif

///////////////////////////////////////
// Preinitialised arrays
///////////////////////////////////////

#ifdef VGPU
#include "arch/avx512/static_data.h" // 64 Byte layout
#endif

#ifdef GEN
#include "arch/sse/static_data.h"
#endif

#ifdef SSE4
#include "arch/sse/static_data.h"
#endif

#if defined(AVX1) || defined (AVXFMA) || defined(AVX2) || defined(AVXFMA4)
#include "arch/avx/static_data.h"
#endif

//#ifdef AVX512
//#include "arch/avx512/static_data.h"
//#endif

#ifdef RRII
#ifndef SVE
#ifndef AVX512
#include "arch/gen64/static_data.h"
#endif
#endif
#endif

#ifdef RIRI
#ifndef SVE
#ifndef AVX512
#include "arch/gen64/static_data.h"
#endif
#endif
#endif

#ifdef LIKWID_PERFMON
	#include <likwid.h>
#endif

// read CPU frequency from file
double read_freq() {

  double freq = 1.0;
  std::ifstream ifile("freq.txt", std::ios::in);

  if (!ifile.is_open()) {
    std::cout << "There was a problem opening freq.txt!\n";
    exit(1);
  }

  ifile >> freq;
  ifile.close();

  return freq;
}

#define  FMT std::dec
int main(int argc, char* argv[])
{
  ////////////////////////////////////////////////////////////////////
  // Option 2: retrieve data from file
  ////////////////////////////////////////////////////////////////////

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#endif

  std::cout << "Usage: bench.* [<replicas=1>] [<iterations=1000>] [psi PF dist L1] [next psi PF dist L2] [next U PF dist L2]" << std::endl << std::endl;

  double frequency = read_freq();

  int latt4[4] = {0};
  int Ls = 1;
  uint64_t nsite = 1;
  uint64_t nreplica = 1;
  uint64_t nbrmax;
  uint64_t umax;
  uint64_t fmax;
  uint64_t vol;

#ifdef RIRI
  FILE *fp = std::fopen("data_riri.bin", "r");
#else
  FILE *fp = std::fopen("data_rrii.bin", "r");
#endif
  if (!fp) {
    std::cout << "There was a problem opening data file!\n";
    exit(1);
  }
  std::fread((void*)&nsite, sizeof(uint64_t), 1, fp);
  std::fread((void*)&latt4[0], sizeof(int), 4, fp);
  std::fread((void*)&Ls, sizeof(int), 1, fp);
  nbrmax = nsite*Ls*8;
  umax   = nsite*18*8 *vComplexD::Nsimd();
  fmax   = nsite*24*Ls*vComplexD::Nsimd();
  vol    = nsite*Ls*vComplexD::Nsimd();
  double* U_static = (double*)malloc(umax * sizeof(double));
  std::fread((void*)U_static, sizeof(double), umax, fp);
  double* Phi_static = (double*)malloc(fmax * sizeof(double));
  std::fread((void*)Phi_static, sizeof(double), fmax, fp);
  double* Psi_cpp_static = (double*)malloc(fmax * sizeof(double));
  std::fread((void*)Psi_cpp_static, sizeof(double), fmax, fp);
  uint64_t* nbr_static = (uint64_t*)malloc(nbrmax * sizeof(uint64_t));
  std::fread((void*)nbr_static, sizeof(uint64_t), nbrmax, fp);
  uint8_t* prm_static = (uint8_t*)malloc(nbrmax * sizeof(uint8_t));
  std::fread((void*)prm_static, sizeof(uint8_t), nbrmax, fp);
  fclose(fp);

  if ((Ls <= 0) && (nsite <= 1)) {
    std::cout << "Illegal Ls and nsite!\n";
    exit(1);
  }

  std::cout << std::endl;

  nreplica = argc > 1 ? atoi(argv[1]) : 1;
  int nrep = argc > 2 ? atoi(argv[2]) : 1000;
  int psi_pf_dist_L1 = argc > 3 ? atoi(argv[3]) : 1;
  int psi_pf_dist_L2 = argc > 4 ? atoi(argv[4]) : 4;
  int u_pf_dist_L2   = argc > 5 ? atoi(argv[5]) : 0;


  // check iterations
  assert(nrep > 0);
  // check if nreplica is > 0 and power of 2
  assert(nreplica > 0);
  assert( (nreplica & (nreplica - 1)) == 0 );

  // near optimal PF measured using SVETemplate7.h and GCC
  if (argc <= 3) {
  /*  not needed anymore
   *  for RRII version use 1 4 0
   *  for RIRI version use 2 4 0
   *  std::cout << "Auto-picking PF distances (tuned for rrii, 32 replicas, nsite 512, 12 threads, FX700 using SVETemplate7.h / GCC 10.1.1)" << std::endl;
    switch(nreplica) {
      case 1:  psi_pf_dist_L1 = 2;
               psi_pf_dist_L2 = 0;
               u_pf_dist_L2   = -3;
               break;
      default:
               psi_pf_dist_L1 = 3;
               psi_pf_dist_L2 = 0;
               u_pf_dist_L2   = -4;
    }*/
  } else {
    std::cout << "User-defined PF distances" << std::endl;
  }

  int threads = 1;
//#ifdef _OPENMP
#ifdef OMP
threads = omp_thread_count();
#endif

  std::cout << "Frequency  = " << frequency << " GHz" << std::endl;
  std::cout << "Threads    = " << threads << std::endl;
  std::cout << "Nsimd      = " << vComplexD::Nsimd() << std::endl;
  std::cout << "Replicas   = " << nreplica << std::endl;
  std::cout << "Iterations = " << nrep << std::endl;

  std::cout << std::endl;

  std::cout << "     psi PF dist L1 = " << psi_pf_dist_L1 << std::endl;
  std::cout << "next psi PF dist L2 = " << psi_pf_dist_L2 << std::endl;
  std::cout << "next   U PF dist L2 = " << u_pf_dist_L2   << std::endl;

  std::cout << std::endl;

  std::cout << "  Latt4    = " << latt4[0] << " x " << latt4[1] << " x " << latt4[2] << " x " << latt4[3] << std::endl;
  std::cout << "  Ls       = " << Ls << std::endl;
  std::cout << "  nsite    = " << nsite * nreplica << std::endl;
  std::cout << "  volume   = " << vol * nreplica << std::endl;
  std::cout << "           = " << latt4[0] << " x " << latt4[1] << " x " << latt4[2] << " x " << latt4[3] << " x " << Ls << std::endl;
  // decompose
  /*
  int Latt[5] = {1,1,1,1,Ls};
  int j = 0;
  auto v = nsite * vComplexD::Nsimd() * nreplica;
  while(v > 1) {
    v /= 2;
    Latt[j % 4] *= 2;
    j++;
  }
  std::cout << "           = " << Latt[3] << " x " << Latt[2] << " x " << Latt[1] << " x " << Latt[0] << " x " << Latt[4] << std::endl;
*/
  std::cout << std::endl;

  uint64_t udata = umax * sizeof(double) * nreplica;
  uint64_t fdata = fmax * sizeof(double) * nreplica;
  std::cout << "  U        = " << udata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  psi in   = " << fdata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  psi out  = " << fdata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  total    = " << (udata + 2 * fdata) / (1024. * 1024.) << " MiB" << std::endl;

  std::cout << std::endl;

  std::cout << "Grid reference benchmark: " << std::endl;

  if (Ls > 1) {
    std::cout << "srun -n 1 ./Benchmark_dwf --mpi 1.1.1.1 --grid "
      << latt4[0] << "." << latt4[1] << "." << latt4[2] << "." << latt4[3]
      << " -Ls " << Ls << " --dslash-asm --threads " << threads << " | grep \": mflop/s =\" "
      << std::endl << std::endl;
  } else {
    std::cout << "srun -n 1 ./Benchmark_wilson --mpi 1.1.1.1 --grid "
      << latt4[0] << "." << latt4[1] << "." << latt4[2] << "." << latt4[3]
      << " --dslash-asm --threads " << threads << " | grep \": mflop/s =\" "
      << std::endl << std::endl;
  }


  /*
  std::FILE *fp;

  Vector<double>   U_static(umax);
  std::fopen("static_data.U.dat", "rb");
  std::fread(&U_static[0]; sizeof(double), umax, fp);
  std::fclose(fp);
  */

  Vector<double>   U(umax*nreplica);
  Vector<double>   Psi(fmax*nreplica);
  Vector<double>   Phi(fmax*nreplica);
  Vector<double>   Psi_cpp(fmax*nreplica);
  Vector<uint64_t> nbr(nsite*Ls*8*nreplica);
  Vector<uint8_t>  prm(nsite*Ls*8*nreplica);

  // enable compute on full chip respecting first touch
  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<U.size()  ;++i) {
    U[i]       = 0.;
  }

  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<Psi.size();++i) {
    Psi[i]     = 0.;
    Phi[i]     = 0.;
    Psi_cpp[i] = 0.;
  }

  #pragma omp parallel for schedule(static)
  for(size_t i=0;i<nbr.size();++i) {
    nbr[i]     = 0;
    prm[i]     = 0;
  }

  // replicas
  for(int replica=0;replica<nreplica;replica++){
    int u=replica*umax;
    int f=replica*fmax;
    int n=replica*nbrmax;
    bcopy(U_static,&U[u],umax*sizeof(double));
    bzero(&Psi[f],fmax*sizeof(double));
    bcopy(nbr_static,&nbr[n],nbrmax*sizeof(uint64_t));
    bcopy(prm_static,&prm[n],nbrmax*sizeof(uint8_t));
    for(int nn=0;nn<nbrmax;nn++){
      nbr[nn+n]+=nsite*Ls*replica; // Shift the neighbour indexes to point to this replica
    }
  }

  Vector<float>   fU(umax*nreplica);
  Vector<float>   fPsi(fmax*nreplica);
  Vector<float>   fPhi(fmax*nreplica);
  Vector<float>   fPsi_cpp(fmax*nreplica);

  //std::cout << "&U   = " << &U[0] << std::endl;
  //std::cout << "&Psi = " << &Psi[0] << std::endl;
  //std::cout << "&Phi = " << &Phi[0] << std::endl;

// Fails for AVX512 !??
//  assert(vComplexD::Nsimd()==EXPAND_SIMD);
  //assert(vComplexF::Nsimd()==EXPAND_SIMD);
  std::cout << "vComplexD::Nsimd() = " << vComplexD::Nsimd() << std::endl;
  std::cout << "DATA_SIMD          = " << DATA_SIMD << std::endl;
  std::cout << "EXPAND_SIMD        = " << EXPAND_SIMD << std::endl;

  const int Nsimd  = EXPAND_SIMD;
  const int NNsimd = DATA_SIMD;
  const int nsimd_replica=Nsimd/NNsimd;
  std::cout << " Expanding SIMD width by "<<nsimd_replica<<"x"<<std::endl;
#ifdef RRII
#define VEC_IDX(ri,n,nn) (ri*Nsimd+nn*NNsimd+n)
#else
#define VEC_IDX(ri,n,nn) (nn*NNsimd*2 + n*2 +ri)
#endif
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite;ss++){
  for(uint32_t s=0;s<Ls;s++){
  for(uint32_t sc=0;sc<12;sc++){
    for(uint32_t n=0;n<NNsimd;n++){
    for(uint32_t nn=0;nn<nsimd_replica;nn++){
      for(uint32_t ri=0;ri<2;ri++){
	int idx = ss*Ls*24*NNsimd
	        +     s*24*NNsimd
    	        +     sc*2*NNsimd;
	int ridx= idx*nsimd_replica+r*nsite*Ls*24*Nsimd;
	Phi     [ridx + VEC_IDX(ri,n,nn) ] =     Phi_static[idx + n*2 + ri];
	Psi_cpp [ridx + VEC_IDX(ri,n,nn) ] = Psi_cpp_static[idx + n*2 + ri];
	fPhi    [ridx + VEC_IDX(ri,n,nn) ] =     Phi_static[idx + n*2 + ri];
	fPsi_cpp[ridx + VEC_IDX(ri,n,nn) ] = Psi_cpp_static[idx + n*2 + ri];
      }
    }}
  }}}}
  std::cout << "Remapped Spinor data\n" ;
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite*9*8;ss++){
    for(uint32_t n=0;n<NNsimd;n++){
    for(uint32_t nn=0;nn<nsimd_replica;nn++){
      for(uint32_t ri=0;ri<2;ri++){
	U [r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + VEC_IDX(ri,n,nn) ] = U_static[ss*NNsimd*2 + n*2 + ri];
	fU[r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + VEC_IDX(ri,n,nn) ] = U_static[ss*NNsimd*2 + n*2 + ri];
      }
    }}
  }}
  std::cout << "Remapped Gauge data\n";

  std::cout << std::endl;
  std::cout << "Calling dslash_kernel "<<std::endl;


  double flops = 1320.0*vol*nreplica;
  //int nrep=1000; // cache warm

#ifdef LIKWID_PERFMON
#pragma omp parallel
  {
	  LIKWID_MARKER_REGISTER("dslash_kernel");
  }
#pragma omp parallel
  {
	  LIKWID_MARKER_START("dslash_kernel");
  }
#endif

#ifdef DOUBLE
  double usec;
  usec = dslash_kernel<vComplexD>(nrep,
                           (vComplexD *)&U[0],
                           (vComplexD *)&Psi[0],
                           (vComplexD *)&Phi[0],
                           &nbr[0],
                           nsite*nreplica,
                           Ls,
                           &prm[0],
			   psi_pf_dist_L1,
			   psi_pf_dist_L2,
			   u_pf_dist_L2);
#else
  double usec = dslash_kernel<vComplexF>(nrep,
			   (vComplexF *)&fU[0],
			   (vComplexF *)&fPsi[0],
			   (vComplexF *)&fPhi[0],
			   &nbr[0],
			   nsite*nreplica,
			   Ls,
			   &prm[0]);

  // Copy back to double
  for(uint64_t i=0; i<fmax*nreplica;i++){
    Psi[i]=fPsi[i];
  }
#endif

#ifdef LIKWID_PERFMON
#pragma omp parallel
  {
	  LIKWID_MARKER_STOP("dslash_kernel");
  }
#endif


  std::cout << std::endl;
#ifdef DOUBLE

  // 8 * 12 + 8 * 9 in
  // 12 out
  double sec = usec / 1000000.;

  uint64_t total_data  = udata + 2 * fdata;
  double tp10          = ((total_data * nrep) / sec) / (1000. * 1000. * 1000.);
  double tp2           = ((total_data * nrep) / sec) / (1024. * 1024. * 1024.);
  // should parametrize % peak by vector size in hardware!
  double percent_peak  = 100. * ((nrep*flops/usec/1000.)/frequency/threads) / (2.*2.*8);
  double cycles        = sec * frequency * 1000. * 1000. * 1000.;
  double gflops_per_s  = nrep*flops/usec/1000.;
  double usec_per_Ls   = usec/nrep/(nsite* nreplica)/Ls;
  double cycles_per_Ls = cycles/nrep/(nsite* nreplica)/Ls;

  std::cout <<"XX\t"<< gflops_per_s << " GFlop/s DP; kernel per vector site "
    << usec_per_Ls <<" usec / " << cycles_per_Ls << " cycles" <<std::endl;

  std::cout <<"YY\t"<< gflops_per_s/frequency << " Flops/cycle DP; kernel per vector site "
    << usec_per_Ls <<" usec / " << cycles_per_Ls << " cycles" <<std::endl;

  std::cout <<"ZZ\t"<< gflops_per_s/frequency/threads << " Flops/cycle DP per thread; kernel per vector site "
    << usec_per_Ls * threads <<" usec / " << cycles_per_Ls * threads << " cycles" <<std::endl;

  std::cout <<std::endl;

  std::cout <<"XX\t"<< gflops_per_s/frequency/threads/ EXPAND_SIMD << " Flops/cycle DP per thread; kernel per single site "
    << usec_per_Ls * threads/ EXPAND_SIMD << " usec / "
    << cycles_per_Ls * threads / EXPAND_SIMD << " cycles" <<std::endl;

  std::cout << std::endl;

  std::cout <<"\t"<< percent_peak << " % peak" << std::endl;

  std::cout << std::endl;

  total_data = (8 * 9 + 8 * 12 + 12) * Ls * 2 * sizeof(double) * vComplexD::Nsimd() * nsite * nreplica;
  tp10 = ((total_data * nrep) / sec) / (1000. * 1000. * 1000.);
  tp2  = ((total_data * nrep) / sec) / (1024. * 1024. * 1024.);
  std::cout <<"\t"<< tp10 << "  GB/s RF throughput (base 10)" <<std::endl;
  std::cout <<"\t"<< tp2  << " GiB/s RF throughput (base  2)" <<std::endl;
  std::cout << "\tdata transfer RF per iteration = " << total_data / (1024. * 1024) << " MiB" << std::endl;

  std::cout << std::endl;

  total_data = (8 * 9 + (8 * 12 + 12) * Ls) * 2 * sizeof(double) * vComplexD::Nsimd() * nsite * nreplica;
  tp10 = ((total_data * nrep) / sec) / (1000. * 1000. * 1000.);
  tp2  = ((total_data * nrep) / sec) / (1024. * 1024. * 1024.);
  std::cout <<"\t"<< tp10 << "  GB/s eff. memory throughput (base 10)" <<std::endl;
  std::cout <<"\t"<< tp2  << " GiB/s eff. memory throughput (base  2)" <<std::endl;
  std::cout << "\teff. data transfer memory per iteration = " << total_data / (1024. * 1024) << " MiB" << std::endl;

  std::cout << std::endl;

  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in double precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#else
  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in single precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#endif
  std::cout << std::endl;

  // One liner result output
  std::cout << "# Threads     Replicas    Volume    GFlop/s    % peak    Cycles per single site    Cycles per vector site    psi PF dist L1    next psi PF dist L2    next U PF dist L2" << std::endl;
  std::cout
    << threads       << "  "
    << nreplica      << "  "
    << latt4[0] << "x" << latt4[1] << "x" << latt4[2] << "x" << latt4[3] << "x" << Ls << "  "
    << gflops_per_s  << "  "
    << percent_peak  << "  "
    << cycles_per_Ls * threads / EXPAND_SIMD << "  "
    << cycles_per_Ls * threads << "  "
    << psi_pf_dist_L1 << "  "
    << psi_pf_dist_L2 << "  "
    << u_pf_dist_L2 << "  "
#ifdef RIRI
    << " RIRI "
#else
    << " RRII "
#endif
    << "XX1" << std::endl << std::endl;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif

  // Check results
  vComplexD *Psi_p = (vComplexD *) &Psi[0];
  vComplexD *Psi_cpp_p = (vComplexD *) &Psi_cpp[0];
  double err=0;
  for(uint64_t r=0; r<nreplica;r++){
    double nref=0;
    double nres=0;
    for(uint64_t ii=0; ii<fmax;ii++){
      uint64_t i=ii+r*fmax;
      err += (Psi_cpp[i]-Psi[i])*(Psi_cpp[i]-Psi[i]);
      nres += Psi[i]*Psi[i];
      nref += Psi_cpp[i]*Psi_cpp[i];
    };
    if (r == 0) {
      std::cout<< "normdiff "<< err<< " ref "<<nref<<" result "<<nres<<std::endl;
    }
    for(int ii=0;ii<64;ii++){
      uint64_t i=ii+r*fmax;
      //std::cout<< i<<" ref "<<Psi_cpp[i]<< " result "<< Psi[i]<<std::endl;
    }
  }
  assert(err <= 1.0e-6);
  return 0;
}
