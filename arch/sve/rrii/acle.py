#!/usr/bin/python3

# SVE templates

# arch
arch = 'A64FX / 512-bit SVE'

# arch header
arch_header = 'arm_sve.h'

# vector length in bytes
arch_vl = 64

# float data type, defines precision
arch_float_type = 'float64_t'

# vector data type
arch_simd_type = f'sv{arch_float_type}'

# address typecast to float
arch_float_typecast = f'({arch_float_type}*)'

# ------------------------------------------------------------------------------

# load op1 from typecast op2 address op3
intrin_load = "{} = svld1(pg1, {}({}))"

# load op1 from typecast op2 address op3 offset op4
intrin_load_offset = "{} = svld1_vnum(pg1, {}({}), (int64_t)({}))"

# pf load
# tbd

# offset pf load
# tbd

# store op3 to typecast op1 address op2
intrin_store = "svst1(pg1, {}({}), {})"

# store op4 to typecast op1 address op2 offset op3
intrin_store_offset = "svst1_vnum(pg1, {}({}), (int64_t)({}), {})"

# pf store
# tbd

# offset pf store
# tbd

# op1 = scatter scalar typecast op2 op3
intrin_dup  = "{} = svdup(({})({}))"

# copy op1 = op2
intrin_mov  = "{} = {}"

# negate op1 = -op2
intrin_neg  = "{} = svneg_x(pg1, {})"

# add op1 = op2 + op3
intrin_add  = "{} = svadd_x(pg1, {}, {})"

# sub op1 = op2 - op3
intrin_sub  = "{} = svsub_x(pg1, {}, {})"

# mul op1 = op2 * op3
intrin_mul  = "{} = svmul_x(pg1, {}, {})"

# fma op1 = op2 + op3 * op4
intrin_fma  = "{} = svmla_x(pg1, {}, {}, {})"

# fms op1 = op2 - op3 * op4
intrin_fms  = "{} = svmls_x(pg1, {}, {}, {})"
