#!/usr/bin/env sh
set -e

./build/tools/caffe train --solver=models/bupt_wangnet/solver.prototxt
