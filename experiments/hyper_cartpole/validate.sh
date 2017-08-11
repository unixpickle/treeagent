#!/bin/bash

if [ ! -f model_out ]; then
  echo 'missing model_out file (hint: run train.sh first)' >&2
  exit 1
fi

neurocli train \
  -batch 10000 \
  -cost mse \
  -net model_out \
  -samples data/test.txt \
  -step 0 \
  -stopsamples 1
