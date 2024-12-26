#!/bin/bash
export PYTHONPATH=$(dirname "$0")/../:$PYTHONPATH
export HF_ENDPOINT="https://hf-mirror.com"
export WORKDIR=$(dirname "$0")/../
source $WORKDIR/bin/env.sh
# 获取所有的参数
args=("$@")
$PYTHON_HOME/python cad_fr/parse.py $@