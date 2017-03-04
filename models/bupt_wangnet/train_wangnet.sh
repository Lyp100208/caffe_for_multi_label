#!/usr/bin/env sh
set -e
DATE=$(date +%Y-%m-%d-%H-%M-%s)
LOG_NAME=bupt_wangnet-${DATE}.log
LOG_DIR=$CAFFE_ROOT/log/$LOG_NAME
caffe train -solver $CAFFE_ROOT/models/bupt_wangnet/solver.prototxt 2>&1 | tee $LOG_DIR
