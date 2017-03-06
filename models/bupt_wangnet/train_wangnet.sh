#!/usr/bin/env sh
set -e
DATE=$(date +%Y-%m-%d-%H-%M-%s)
LOG_NAME=bupt_wangnet-${DATE}.log
LOG_DIR=$CAFFE_M_ROOT/log/$LOG_NAME
TOOLS_ROOT=$CAFFE_M_ROOT/build/tools
$TOOLS_ROOT/caffe train -solver $CAFFE_M_ROOT/models/bupt_wangnet/solver.prototxt 2>&1 | tee $LOG_DIR
