    /*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid 

    Source file: ./lib/simd/Grid_sse4.h

    Copyright (C) 2015

Author: Azusa Yamaguchi <ayamaguc@staffmail.ed.ac.uk>
Author: Peter Boyle <paboyle@ph.ed.ac.uk>
Author: neo <cossu@post.kek.jp>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along
    with this program; if not, write to the Free Software Foundation, Inc.,
    51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

    See the full license in the file "LICENSE" in the top level distribution directory
    *************************************************************************************/
    /*  END LEGAL */
//----------------------------------------------------------------------
/*! @file Grid_sse4.h
  @brief Optimization libraries for SSE4 instructions set

  Using intrinsics
*/
// Time-stamp: <2015-06-16 23:27:54 neo>
//----------------------------------------------------------------------

#include <pmmintrin.h>


  template<class vtype>
  union uconv {
    __m128 f;
    vtype v;
  };

  union u128f {
    __m128 v;
    float f[4];
  };
  union u128d {
    __m128d v;
    double f[2];
  };
  
  struct Vsplat{
    //Complex float
    inline __m128 operator()(float a, float b){
      return _mm_set_ps(b,a,b,a);
    }
    // Real float
    inline __m128 operator()(float a){
      return _mm_set_ps(a,a,a,a);
    }
    //Complex double
    inline __m128d operator()(double a, double b){
      return _mm_set_pd(b,a);
    }
    //Real double
    inline __m128d operator()(double a){
      return _mm_set_pd(a,a);
    }
    //Integer
    inline __m128i operator()(Integer a){
      return _mm_set1_epi32(a);
    }
  };

  struct Vstore{
    //Float 
    inline void operator()(__m128 a, void* F){
      _mm_store_ps((float *)F,a);
    }
    //Double
    inline void operator()(__m128d a, void* D){
      _mm_store_pd((double *)D,a);
    }
    //Integer
    inline void operator()(__m128i a, void* I){
      _mm_store_si128((__m128i *)I,a);
    }

  };

  struct Vstream{
    //Float
    inline void operator()(float * a, __m128 b){
      _mm_stream_ps(a,b);
    }
    //Double
    inline void operator()(double * a, __m128d b){
      _mm_stream_pd(a,b);
    }


  };

  struct Vset{
    // Complex float 
    inline __m128 operator()(ComplexF *a){
      return _mm_set_ps(a[1].imag(), a[1].real(),a[0].imag(),a[0].real());
    }
    // Complex double 
    inline __m128d operator()(ComplexD *a){
      return _mm_set_pd(a[0].imag(),a[0].real());
    }
    // Real float 
    inline __m128 operator()(float *a){
      return _mm_set_ps(a[3],a[2],a[1],a[0]);
    }
    // Real double
    inline __m128d operator()(double *a){
      return _mm_set_pd(a[1],a[0]);
    }
    // Integer
    inline __m128i operator()(Integer *a){
      return _mm_set_epi32(a[3],a[2],a[1],a[0]);
    }


  };

  template <typename Out_type, typename In_type>
  struct Reduce{
    //Need templated class to overload output type
    //General form must generate error if compiled
    inline Out_type operator()(In_type in){
      printf("Error, using wrong Reduce function\n");
      exit(1);
      return 0;
    }
  };

  /////////////////////////////////////////////////////
  // Arithmetic operations
  /////////////////////////////////////////////////////
  struct Sum{
    //Complex/Real float
    inline __m128 operator()(__m128 a, __m128 b){
      return _mm_add_ps(a,b);
    }
    //Complex/Real double
    inline __m128d operator()(__m128d a, __m128d b){
      return _mm_add_pd(a,b);
    }
    //Integer
    inline __m128i operator()(__m128i a, __m128i b){
      return _mm_add_epi32(a,b);
    }
  };

  struct Sub{
    //Complex/Real float
    inline __m128 operator()(__m128 a, __m128 b){
      return _mm_sub_ps(a,b);
    }
    //Complex/Real double
    inline __m128d operator()(__m128d a, __m128d b){
      return _mm_sub_pd(a,b);
    }
    //Integer
    inline __m128i operator()(__m128i a, __m128i b){
      return _mm_sub_epi32(a,b);
    }
  };

  struct MultRealPart{
    inline __m128 operator()(__m128 a, __m128 b){
      __m128 ymm0;
      ymm0  = _mm_shuffle_ps(a,a,_MM_SELECT_FOUR_FOUR(2,2,0,0)); // ymm0 <- ar ar,
      return  _mm_mul_ps(ymm0,b);                       // ymm0 <- ar bi, ar br
    }
    inline __m128d operator()(__m128d a, __m128d b){
      __m128d ymm0;
      ymm0 = _mm_shuffle_pd(a,a,0x0); // ymm0 <- ar ar, ar,ar b'00,00
      return _mm_mul_pd(ymm0,b);      // ymm0 <- ar bi, ar br
    }
  };
  struct MaddRealPart{
    inline __m128 operator()(__m128 a, __m128 b, __m128 c){
      __m128 ymm0 =  _mm_shuffle_ps(a,a,_MM_SELECT_FOUR_FOUR(2,2,0,0)); // ymm0 <- ar ar,
      return _mm_add_ps(_mm_mul_ps( ymm0, b),c);                         
    }
    inline __m128d operator()(__m128d a, __m128d b, __m128d c){
      __m128d ymm0 = _mm_shuffle_pd( a, a, 0x0 );
      return _mm_add_pd(_mm_mul_pd( ymm0, b),c);                         
    }
  };

  struct MultComplex{
    // Complex float
    inline __m128 operator()(__m128 a, __m128 b){
      __m128 ymm0,ymm1,ymm2;
      ymm0 = _mm_shuffle_ps(a,a,_MM_SELECT_FOUR_FOUR(2,2,0,0)); // ymm0 <- ar ar,
      ymm0 = _mm_mul_ps(ymm0,b);                       // ymm0 <- ar bi, ar br
      ymm1 = _mm_shuffle_ps(b,b,_MM_SELECT_FOUR_FOUR(2,3,0,1)); // ymm1 <- br,bi
      ymm2 = _mm_shuffle_ps(a,a,_MM_SELECT_FOUR_FOUR(3,3,1,1)); // ymm2 <- ai,ai
      ymm1 = _mm_mul_ps(ymm1,ymm2);                    // ymm1 <- br ai, ai bi
      return _mm_addsub_ps(ymm0,ymm1);    
    }
    // Complex double
    inline __m128d operator()(__m128d a, __m128d b){
      __m128d ymm0,ymm1,ymm2;
      ymm0 = _mm_shuffle_pd(a,a,0x0);   // ymm0 <- ar ar,
      ymm0 = _mm_mul_pd(ymm0,b);        // ymm0 <- ar bi, ar br
      ymm1 = _mm_shuffle_pd(b,b,0x1);   // ymm1 <- br,bi   b01
      ymm2 = _mm_shuffle_pd(a,a,0x3);   // ymm2 <- ai,ai   b11
      ymm1 = _mm_mul_pd(ymm1,ymm2);     // ymm1 <- br ai, ai bi
      return _mm_addsub_pd(ymm0,ymm1);  
    }
  };

  struct Mult{

    inline void mac(__m128 &a, __m128 b, __m128 c){
      a= _mm_add_ps(_mm_mul_ps(b,c),a);
    }

    inline void mac(__m128d &a, __m128d b, __m128d c){
      a= _mm_add_pd(_mm_mul_pd(b,c),a);
    }

    // Real float
    inline __m128 operator()(__m128 a, __m128 b){
      return _mm_mul_ps(a,b);
    }
    // Real double
    inline __m128d operator()(__m128d a, __m128d b){
      return _mm_mul_pd(a,b);
    }
  };

  struct Div{
    // Real float
    inline __m128 operator()(__m128 a, __m128 b){
      return _mm_div_ps(a,b);
    }
    // Real double
    inline __m128d operator()(__m128d a, __m128d b){
      return _mm_div_pd(a,b);
    }
  };


  struct Conj{
    // Complex single
    inline __m128 operator()(__m128 in){
      return _mm_xor_ps(_mm_addsub_ps(_mm_setzero_ps(),in), _mm_set1_ps(-0.f));
    }
    // Complex double
    inline __m128d operator()(__m128d in){
      return _mm_xor_pd(_mm_addsub_pd(_mm_setzero_pd(),in), _mm_set1_pd(-0.f));//untested
    }
    // do not define for integer input
  };

  struct TimesMinusI{
    //Complex single
    inline __m128 operator()(__m128 in, __m128 ret){
      __m128 tmp =_mm_addsub_ps(_mm_setzero_ps(),in); // r,-i
      ret = _mm_shuffle_ps(tmp,tmp,_MM_SELECT_FOUR_FOUR(2,3,0,1));
      return ret;
    }
    //Complex double
    inline __m128d operator()(__m128d in, __m128d ret){
      __m128d tmp =_mm_addsub_pd(_mm_setzero_pd(),in); // r,-i
      ret = _mm_shuffle_pd(tmp,tmp,0x1);
      return ret;
    }


  };

  struct TimesI{
    //Complex single
    inline __m128 operator()(__m128 in, __m128 ret){
      __m128 tmp =_mm_shuffle_ps(in,in,_MM_SELECT_FOUR_FOUR(2,3,0,1));
      ret = _mm_addsub_ps(_mm_setzero_ps(),tmp); // r,-i
      return ret;
    }
    //Complex double
    inline __m128d operator()(__m128d in, __m128d ret){
      __m128d tmp = _mm_shuffle_pd(in,in,0x1);
      ret = _mm_addsub_pd(_mm_setzero_pd(),tmp); // r,-i
      return ret;
    }
  };

  struct Permute{

    static inline __m128 Permute0(__m128 in){
      __m128 tmp = _mm_shuffle_ps(in,in,_MM_SELECT_FOUR_FOUR(1,0,3,2)); //AB CD -> CD AB
      return tmp;
    };
    static inline __m128 Permute1(__m128 in){
      __m128 tmp = _mm_shuffle_ps(in,in,_MM_SELECT_FOUR_FOUR(2,3,0,1)); //AB CD -> BA DC
      return tmp;
    };
    static inline __m128 Permute2(__m128 in){
      return in;
    };
    static inline __m128 Permute3(__m128 in){
      return in;
    };

    static inline __m128d Permute0(__m128d in){ //AB -> BA
      __m128d tmp= _mm_shuffle_pd(in,in,0x1);
      return tmp;
    };
    static inline __m128d Permute1(__m128d in){
      return in;
    };
    static inline __m128d Permute2(__m128d in){
      return in;
    };
    static inline __m128d Permute3(__m128d in){
      return in;
    };
  };

  
#define _my_alignr_epi32(a,b,n) _mm_alignr_epi8(a,b,(n*4)%16)
#define _my_alignr_epi64(a,b,n) _mm_alignr_epi8(a,b,(n*8)%16)

  //////////////////////////////////////////////
  // Some Template specialization


  //Complex float Reduce
  template<>
  inline ComplexF Reduce<ComplexF, __m128>::operator()(__m128 in){
    __m128 v1; // two complex
    v1= Permute::Permute0(in); 
    v1= _mm_add_ps(v1,in);
    u128f conv;    conv.v=v1;
    return ComplexF(conv.f[0],conv.f[1]);
  }
  //Real float Reduce
  template<>
  inline RealF Reduce<RealF, __m128>::operator()(__m128 in){
    __m128 v1,v2; // quad single
    v1= Permute::Permute0(in); 
    v1= _mm_add_ps(v1,in);
    v2= Permute::Permute1(v1); 
    v1 = _mm_add_ps(v1,v2);
    u128f conv; conv.v=v1;
    return conv.f[0];
  }
  
  
  //Complex double Reduce
  template<>
  inline ComplexD Reduce<ComplexD, __m128d>::operator()(__m128d in){
    u128d conv; conv.v = in;
    return ComplexD(conv.f[0],conv.f[1]);
  }
  
  //Real double Reduce
  template<>
  inline RealD Reduce<RealD, __m128d>::operator()(__m128d in){
    __m128d v1;
    v1 = Permute::Permute0(in); 
    v1 = _mm_add_pd(v1,in);
    u128d conv; conv.v = v1;
    return conv.f[0];
  }





//////////////////////////////////////////////////////////////////////////////////////
// Here assign types 

  typedef __m128i SIMD_Htype;  // Single precision type
  typedef __m128  SIMD_Ftype;  // Single precision type
  typedef __m128d SIMD_Dtype; // Double precision type
  typedef __m128i SIMD_Itype; // Integer type
  typedef SIMD_Ftype SIMD_CFtype;  // Single precision type
  typedef SIMD_Dtype SIMD_CDtype; // Double precision type

  // prefetch utilities
  inline void v_prefetch0(int size, const char *ptr){};
  inline void prefetch_HINT_T0(const char *ptr){
    _mm_prefetch(ptr,_MM_HINT_T0);
  }

  // Function name aliases
  typedef Vsplat   VsplatSIMD;
  typedef Vstore   VstoreSIMD;
  typedef Vset     VsetSIMD;
  typedef Vstream  VstreamSIMD;
  template <typename S, typename T> using ReduceSIMD = Reduce<S,T>;

 


  // Arithmetic operations
  typedef Sum         SumSIMD;
  typedef Sub         SubSIMD;
  typedef Div         DivSIMD;
  typedef Mult        MultSIMD;
  typedef MultComplex MultComplexSIMD;
  typedef MultRealPart MultRealPartSIMD;
  typedef MaddRealPart MaddRealPartSIMD;
  typedef Conj        ConjSIMD;
  typedef TimesMinusI TimesMinusISIMD;
  typedef TimesI      TimesISIMD;


