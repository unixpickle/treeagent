#!/bin/bash

if [ ! -f model_out ]; then
  echo 'no model_out file (hint: run train.sh first)' >&2
  exit 1
fi

echo 'Enter depth, step size, and step decay as a space separated list:'
echo

neurocli run -net model_out
