#!/usr/bin/python3

# AVX512 templates

# arch
arch = 'AVX512'

# arch header
arch_header = 'immintrin.h'

# vector length in bytes
arch_vl = 64

# float data type, defines precision
arch_float_type = 'double'

# vector data type
arch_simd_type = f'__m512d'

# address typecast to float
arch_float_typecast = f'({arch_float_type}*)'

# ------------------------------------------------------------------------------

# load op1 from typecast op2 address op3
intrin_load = "{} = _mm512_load_pd({}({}))"

# load op1 from typecast op2 address op3 offset op4
#intrin_load_offset = "{} = svld1_vnum(pg1, {}({}), (int64_t)({}))"
intrin_load_offset = "{} = XXXLOADOFFSET({}({}), (int64_t)({}))"

# pf load
# tbd

# offset pf load
# tbd

# store op3 to typecast op1 address op2
#intrin_store = "_mm512_store_pd({}({}), {})"
intrin_store = "_mm512_stream_pd({}({}), {})"

# store op4 to typecast op1 address op2 offset op3
#intrin_store_offset = "svst1_vnum(pg1, {}({}), (int64_t)({}), {})"
intrin_store_offset = "XXXSTOREOFFSET(pg1, {}({}), (int64_t)({}), {})"

# pf store
# tbd

# offset pf store
# tbd

# op1 = scatter scalar typecast op2 op3
intrin_dup = "{} = _mm512_set1_pd({}({}))"

# copy op1 = op2
intrin_mov  = "{} = {}"

# negate op1 = -op2
intrin_neg  = "{} = _mm512_sub_pd(_mm512_setzero_pd(), {})"

# add op1 = op2 + op3
intrin_add  = "{} = _mm512_add_pd({}, {})"

# sub op1 = op2 - op3
intrin_sub  = "{} = _mm512_sub_pd({}, {})"

# mul op1 = op2 * op3
intrin_mul  = "{} = _mm512_mul_pd({}, {})"

# fma op1 = op2 * op3 + op4
intrin_fma  = "{} = _mm512_fmadd_pd({}, {}, {})"

# fnma op1 = - op2 * op3 + op4
intrin_fnma = "{} = _mm512_fnmadd_pd({}, {}, {})"

# fms op1 = op2 * op3 - op4
intrin_fms  = "{} = _mm512_fmsub_pd({}, {}, {})"

# permute op1 = permute# op2
#intrin_permute = "{} = svtbl({}, {})"
intrin_permute0 = "{} = _mm512_shuffle_f64x2({},{},_MM_SELECT_FOUR_FOUR(1,0,3,2))"
intrin_permute1 = "{} = _mm512_shuffle_f64x2({},{},_MM_SELECT_FOUR_FOUR(2,3,0,1))"
intrin_permute2 = "{} = _mm512_shuffle_pd({},{},0x55)"
intrin_permute3 = "{} = XXXPERMUTE3({},{})"
