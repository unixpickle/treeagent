#!/bin/bash

if [ ! -f model_out ]; then
  neurocli new -in model.txt -out model_out || exit 1
fi

neurocli train \
  -batch 16 \
  -cost mse \
  -net model_out \
  -samples data/train.txt \
  -stopsamples 524288 2>&1 | tail -n 2
