#!/usr/bin/env python

#
# The MIT License (MIT)
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

from __future__ import print_function

import sys
import subprocess
import argparse

SAMPLES_PER_WARP = {
    ('sm_70', False): 24,
    ('sm_80', False): 17,
    ('sm_80', True): 40
}


def main():
    parser = argparse.ArgumentParser(
        description="Compile & run mcmc prob training benchmark")
    parser.add_argument(
        "--no-run",
        action="store_true",
        required=False,
        help="Do not run the compiled program"
    )
    parser.add_argument(
        "--arch",
        required=False,
        default="sm_80",
        help="Compute capability: [sm_70, sm_80]"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        required=False,
        help="If set, produce debug output (also sets the DEBUG macro)"
    )
    parser.add_argument(
        "--tc",
        action="store_true",
        required=False,
        help="If set, use the Tensor-core based kernel rather than FFMA. "
             "Can only be set with arch sm_80"
    )
    args = parser.parse_args()
    cmd = ["nvcc", "-std=c++14", "--extended-lambda", "-Iinclude"]
    if not args.no_run:
        cmd += ["-run"]
    samples_per_warp = SAMPLES_PER_WARP.get((args.arch, args.tc))
    if samples_per_warp is None:
        raise ValueError(
            "Invalid combination of arch ({}) and tensor-core use ({})".format(
                args.arch, args.tc))
    cmd += ["-arch={}".format(args.arch),
            "-DDEVICE_ARCH={}0".format(args.arch[-2:]),
            "-DN_WARP_SAMPLES={}".format(samples_per_warp)]
    if args.tc:
        cmd += ["-DUSE_TC=1"]
    if args.debug:
        cmd += ["-g", "-G"]
    else:
        cmd += ["-O3", "-lineinfo"]
    cmd += ["main.cu"]
    print("Compile/run command: " + " ".join(cmd))
    child_proc = subprocess.Popen(cmd)
    result = child_proc.wait()
    return result


if __name__ == "__main__":
    sys.exit(main())
