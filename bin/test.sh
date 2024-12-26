#!/bin/bash
export PYTHONPATH=$(dirname "$0")/../:$PYTHONPATH
export HF_ENDPOINT="https://hf-mirror.com"
export WORKDIR=$(dirname "$0")/../
source $WORKDIR/bin/env.sh
$PYTHON_HOME/python cad_fr/test.py