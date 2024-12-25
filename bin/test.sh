#!/bin/bash
PYTHON_HOME=/Users/yzzer/miniconda3/envs/cad-fr/bin
export PYTHONPATH=$(pwd):$PYTHONPATH
export HF_ENDPOINT="https://hf-mirror.com"
$PYTHON_HOME/python cad_fr/test.py