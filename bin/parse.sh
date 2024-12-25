#!/bin/bash
PYTHON_HOME=/Users/yzzer/miniconda3/envs/cad-fr/bin
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_ENDPOINT="https://hf-mirror.com"
# 获取所有的参数
args=("$@")
$PYTHON_HOME/python cad_fr/parse.py $@