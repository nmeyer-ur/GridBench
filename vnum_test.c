#include <stdio.h>
#include <arm_sve.h>

const int VL = 8;  // vector fits 8 doubles
const int ARRAY_SIZE = 128 * VL;

int main() {

  float64_t src[ARRAY_SIZE];
  float64_t dst[ARRAY_SIZE];

  // init src array
  for(int i=0; i<ARRAY_SIZE; i++) {
    src[i] = (float64_t)i;
  }

  // copy src to dst using _vnum
  svbool_t pg1 = svptrue_b64();
  svfloat64_t z;

  for(int i=0; i<ARRAY_SIZE / VL; i++) {
    z = svld1_vnum(pg1, src, (int64_t)(i));
    svst1_vnum(pg1, dst, (int64_t)(i), z);
  }

  // check
    printf("#        Vector #    src[#]    dst[#]\n");
  for(int i=0; i<ARRAY_SIZE; i++) {
    printf("%4d   %4d   %8.1lf  %8.1lf  ", i, i / VL, src[i], dst[i]);
    if (src[i] == dst[i]) {
      printf("ok\n");
    } else {
      printf("wrong\n");
      break;
    }
  }

  return 0;
}
