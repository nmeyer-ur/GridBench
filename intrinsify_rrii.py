#!/usr/bin/python3

# RRII Intrinsifier

import re
import sys
from acle import *

# src file
src = 'WilsonKernelsHandCpuSVETemplate.h'

class Emitter:
    """Emitter class generates intrinsics functions calls"""

    def __init__(self):
        self._leading = 4 * ' '     # leading spaces
        self._trailing = ''         # trailing character
        self._collection = []       # intrinsics container
        self._backslash = '\\'      # trailing backslash
        self._bracket = '}'         # trailing bracket
        self._isSpinor = False      # spinor addressing
        self._isGauge = False       # gauge addressing
        self._loadstore_offset = False    # armclang has a problem with load / store with offsets!! Be cautious

    def trailing(self, slash=False, bracket=False):
        self._trailing = ''
        if (slash == True):
            self._trailing = self._backslash
        elif (bracket == True):
            self._trailing = self._bracket

    def addressing(self, Spinor=False, Gauge=False):
        if Spinor == Gauge:
            raise

        if Spinor:
            self._isSpinor = True
            self._isGauge  = False
        else:
            self._isSpinor = False
            self._isGauge  = True

    def re(self, variable):
        """Emit real part"""
        return variable + '_re'

    def im(self, variable):
        """Emit imag part"""
        return variable + '_im'

    def emit(self, line):
        """Parser and emitter calls"""
        #if not any(re.findall(r'result|chi|=', line, re.I)):
        #    print(line, end="")
            #return

        # addressing scheme, not sure if necessary
        if ('SiteSpinor' in line):
            self.addressing(Spinor=True)
        elif ('MULT_2SPIN' in line):
            self.addressing(Gauge=True)

        # pass through
        if not (('result_' in line) or ('Chi_' in line) or ('Chimu_' in line) or ('U_' in line) or ('DEBUG' in line)): # or ('=' in line)):
            print(line, end="")
            return

        # catch aliases
        if ('#define' in line) and ('Chi' in line):
            p = re.compile(r'define (\w+) (\w+)')
            op = p.search(line)
            if op:
                self.trailing()  # no backslash or bracket here
                self.cdefine(op.group(1), op.group(2))
                return

        if ('PREFETCH' in line):
            print(line, end="")
            return

        # other pre-processor stuff
        if ('#' in line):
            print(line, end="")
            return

        # debugging stuff
        if ('XX' in line):
            print(line, end="")
            return

        # type declarations
        p = re.compile(r'Simd (\w+);')
        op = p.search(line)
        if op:
            self.declare_simd(op.group(1))
            return

        # valid for conversion
        #print(line, end="")

        # trailing char
        self.trailing()
        if ('\\' in line):
            self.trailing(slash=True)
        elif ('}' in line):
            self.trailing(bracket=True)

        # remove spaces
        line = line.replace(' ', '')

        # have line
        # expand op1 += ... -> op1 = op1 + ...
        if ('+=' in line):
            tmp = line.split('+=')
            line = f'{tmp[0]}={tmp[0]}+{tmp[1]}'

        # expand op1 -= ... -> op1 = op1 - ...
        if ('-=' in line):
            tmp = line.split('-=')
            line = f'{tmp[0]}={tmp[0]}-{tmp[1]}'

        #print(line)

        # conversions
        # op1 = op2
        p = re.compile(r'(\w+)=(\w+);')
        op = p.search(line)
        if op:
            self.cmov(op.group(1), op.group(2))
            return

        # op1 = op2 + op3
        p = re.compile(r'(\w+)=(\w+)\+(\w+);')
        op = p.search(line)
        if op:
            self.cadd(op.group(1), op.group(2), op.group(3))
            return

        # op1 = op2 - op3
        p = re.compile(r'(\w+)=(\w+)-(\w+);')
        op = p.search(line)
        if op:
            self.csub(op.group(1), op.group(2), op.group(3))
            return

        # op1 = op2 * op3
        p = re.compile(r'(\w+)=(\w+)\*(\w+);')
        op = p.search(line)
        if op:
            self.cmul(op.group(1), op.group(2), op.group(3))
            return

        # op1 = op2 + op3 * op4
        p = re.compile(r'(\w+)=(\w+)\+(\w+)\*(\w+);')
        op = p.search(line)
        if op:
            self.cfma(op.group(1), op.group(2), op.group(3), op.group(4))
            return

        # op1 = timesI op2
        p = re.compile(r'(\w+)=timesI\((\w+)\);')
        op = p.search(line)
        if op:
            self.cmovTimesI(op.group(1), op.group(2))
            return

        # op1 = op2 + timesI op3
        p = re.compile(r'(\w+)=(\w+)\+timesI\((\w+)\);')
        op = p.search(line)
        if op:
            self.caddTimesI(op.group(1), op.group(2), op.group(3))
            return

        # op1 = op2 - timesI op3
        p = re.compile(r'(\w+)=(\w+)-timesI\((\w+)\);')
        op = p.search(line)
        if op:
            self.csubTimesI(op.group(1), op.group(2), op.group(3))
            return

        # op1 = timesMinusI op2
        p = re.compile(r'(\w+)=timesMinusI\((\w+)\);')
        op = p.search(line)
        if op:
            self.cmovTimesMinusI(op.group(1), op.group(2))
            return

        # Read
        p = re.compile(r'(\w+)=.*Read\(ref\[(\d+)\]\[(\d+)\]', re.I)
        op = p.search(line)
        if op:
            self.cRead(op.group(1), op.group(2), op.group(3))
            return

        # Write
        p = re.compile(r'Write\(ref\[(\d+)\]\[(\d+)\],(\w+),', re.I)
        op = p.search(line)
        if op:
            #print(op.group(0), op.group(1), op.group(2), op.group(3))
            self.cWrite(op.group(3), op.group(1), op.group(2))
            return

        # done, pass through SYCL stuff
        if ('Permute' in line):
            print(line, end="")
            return

        print('---')
        print('Leftover detected:')
        print(line)
        raise

    def _emit(self):
        """Emit intrinsics"""
        #print(self._collection)

        instructions = len(self._collection)
        if instructions == 1:
            line = self._collection[0]
            x = f"{self._leading}{line};{self._trailing}"
            print(x)
        else:
            for line in self._collection[0:-1]:
                x = f"{self._leading}{line};{self._backslash}"
                print(x)
            line = self._collection[-1]
            x = f"{self._leading}{line};{self._trailing}"
            print(x)

        self._collection = []

    def _emitdefine(self):
        """Emit define"""
        for line in self._collection:
            x = f"{line}"
            print(x)

        self._collection = []

    def _collect(self, line):
        """Collect intrinsic"""
        self._collection.append(line)
        #print()

    def cdefine(self, op1, op2):
        """#define statements for operand aliasing"""
        r = f'#define {self.re(op1)} {self.re(op2)}'
        i = f'#define {self.im(op1)} {self.im(op2)}'
        self._collect(r)
        self._collect(i)
        self._emitdefine()
        #print()

    def declare_simd(self, op):
        r = f'{arch_simd_type} {self.re(op)}'
        i = f'{arch_simd_type} {self.im(op)}'
        self._collect(r)
        self._collect(i)
        self._emit()

    def cmov(self, op1, op2):
        """Emit complex mov
           op1 = op2"""
        # ok
        r = intrin_mov.format(self.re(op1), self.re(op2))
        i = intrin_mov.format(self.im(op1), self.im(op2))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cadd(self, op1, op2, op3):
        """Emit complex add
           op1 = op2 + op3"""
        # ok
        r = intrin_add.format(self.re(op1), self.re(op2), self.re(op3))
        i = intrin_add.format(self.im(op1), self.im(op2), self.im(op3))
        self._collect(r)
        self._collect(i)
        self._emit()

    def csub(self, op1, op2, op3):
        """Emit complex sub
           op1 = op2 - op3"""
        # ok
        r = intrin_sub.format(self.re(op1), self.re(op2), self.re(op3))
        i = intrin_sub.format(self.im(op1), self.im(op2), self.im(op3))
        self._collect(r)
        self._collect(i)
        self._emit()

    def caddTimesI(self, op1, op2, op3):
        """Emit complex add times I
           op1 = op2 + timesI op 3"""
        # ok
        r = intrin_sub.format(self.re(op1), self.re(op2), self.im(op3))
        i = intrin_add.format(self.im(op1), self.im(op2), self.re(op3))
        self._collect(r)
        self._collect(i)
        self._emit()

    def csubTimesI(self, op1, op2, op3):
        """Emit complex sub times I
           op1 = op2 - timesI op3"""
        # ok
        r = intrin_add.format(self.re(op1), self.re(op2), self.im(op3))
        i = intrin_sub.format(self.im(op1), self.im(op2), self.re(op3))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cmovTimesI(self, op1, op2):
        """Emit complex mov times I
           op1 = timesI op2"""
        # ok
        r = intrin_neg.format(self.re(op1), self.im(op2))
        i = intrin_mov.format(self.im(op1), self.re(op2))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cmovTimesMinusI(self, op1, op2):
        """Emit complex mov timesMinusI
           op1 = timesMinusI op2"""
        # ok
        r = intrin_mov.format(self.re(op1), self.im(op2))
        i = intrin_neg.format(self.im(op1), self.re(op2))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cmul(self, op1, op2, op3):
        """Emit complex mul
           op1 = op2 * op3"""
        # ok
        # rr-ii
        # ri+ir
        r = intrin_mul.format(self.re(op1), self.re(op2), self.re(op3))
        i = intrin_mul.format(self.im(op1), self.re(op2), self.im(op3))
        self._collect(r)
        self._collect(i)
        r = intrin_fms.format(self.re(op1), self.re(op1), self.im(op2), self.im(op3))
        i = intrin_fma.format(self.im(op1), self.im(op1), self.im(op2), self.re(op3))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cfma(self, op1, op2, op3, op4):
        """Emit complex fma
           op1 = op2 + op3 * op4"""
        # r + rr-ii
        # i + ri+ir
        # ok
        r = intrin_fma.format(self.re(op1), self.re(op2), self.re(op3), self.re(op4))
        i = intrin_fma.format(self.im(op1), self.im(op2), self.re(op3), self.im(op4))
        self._collect(r)
        self._collect(i)
        r = intrin_fms.format(self.re(op1), self.re(op1), self.im(op3), self.im(op4))
        i = intrin_fma.format(self.im(op1), self.im(op1), self.im(op3), self.re(op4))
        self._collect(r)
        self._collect(i)
        self._emit()

    def cRead(self, op1, row, col):
        """Emit complex load"""
        displacement = 0

        # offset = 0..7 works with armclang; gcc and fcc are always fine
        if self._loadstore_offset:
            cl_offset = 2 * 3 * int(row) + 2 * int(col)
            displacement = 8 * (cl_offset // 8)
            base = f'base + {arch_vl} * {displacement}'

        if self._isGauge:
            cl_offset = 2 * 3 * int(row) + 2 * int(col) - displacement
            off_r  = f'{cl_offset}'
            addr_r = f'base + {arch_vl} * ({off_r})'
            off_i  = f'{cl_offset + 1}'
            addr_i = f'base + {arch_vl} * ({off_i})'
        else:
            cl_offset = 2 * 3 * int(row) + 2 * int(col) - displacement
            off_r  = f'{cl_offset}'
            addr_r = f'base + {arch_vl} * ({off_r})'
            off_i  = f'{cl_offset + 1}'
            addr_i = f'base + {arch_vl} * ({off_i})'

        #print(addr_r)
        if self._loadstore_offset:
            r = intrin_load_offset.format(self.re(op1), arch_float_typecast, base, off_r)
            i = intrin_load_offset.format(self.im(op1), arch_float_typecast, base, off_i)
        else:
            r = intrin_load.format(self.re(op1), arch_float_typecast, addr_r)
            i = intrin_load.format(self.im(op1), arch_float_typecast, addr_i)

        self._collect(r)
        self._collect(i)
        self._emit()

    def cWrite(self, op1, row, col):
        """Emit complex store"""
        displacement = 0

        # offset = 0..7 works with armclang; gcc and fcc are always fine
        if self._loadstore_offset:
            cl_offset = 2 * 3 * int(row) + 2 * int(col)
            displacement = 8 * (cl_offset // 8)
            base = f'base + {arch_vl} * {displacement}'

        if self._isGauge:
            cl_offset = 2 * 3 * int(row) + 2 * int(col) - displacement
            off_r  = f'{cl_offset}'
            addr_r = f'base + {arch_vl} * ({off_r})'
            off_i  = f'{cl_offset + 1}'
            addr_i = f'base + {arch_vl} * ({off_i})'
        else:
            cl_offset = 2 * 3 * int(row) + 2 * int(col) - displacement
            off_r  = f'{cl_offset}'
            addr_r = f'base + {arch_vl} * ({off_r})'
            off_i  = f'{cl_offset + 1}'
            addr_i = f'base + {arch_vl} * ({off_i})'

        #print(addr_r)
        if self._loadstore_offset:
            r = intrin_store_offset.format(arch_float_typecast, base, off_r, self.re(op1))
            i = intrin_store_offset.format(arch_float_typecast, base, off_i, self.im(op1))
        else:
            r = intrin_store.format(arch_float_typecast, addr_r, self.re(op1))
            i = intrin_store.format(arch_float_typecast, addr_i, self.im(op1))

        self._collect(r)
        self._collect(i)
        self._emit()

# end Class Emitter


# main
if __name__ == "__main__":

    #print(len(sys.argv))

    if (len(sys.argv) != 2):
        print(f"Usage: {sys.argv[0]} <src file>")
        print("Missing source file. Exiting.")
        sys.exit(1)

    src = sys.argv[1]

    try:
        file1 = open(src, 'r')
    except FileNotFoundError:
        print(f"File {src} not accessible. Exiting")
        sys.exit(1)

    content = file1.readlines()
    file1.close()

    header = f"""\
// intrinsify
// arch                {arch}
// vector length       {arch_vl} bytes, {arch_vl * 8} bits
// header file         {arch_header}
// float type          {arch_float_type}
// float* typecast     {arch_float_typecast}
// simd type           {arch_simd_type}

#include <{arch_header}>
    """

    print(header)

    emitter = Emitter()
    for line in content:
        emitter.emit(line)
