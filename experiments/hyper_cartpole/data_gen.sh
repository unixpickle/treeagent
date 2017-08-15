#!/bin/bash

mkdir data || exit 1

cat search_output.csv |
  tail -n +2 |
  sed -E $'s/,([-0-9\\.]*)$/\\\n\\1/g' |
  tr ',' ' ' >data/all.txt

cat data/all.txt | tail -n +501 >data/train.txt
cat data/all.txt | head -n 500 >data/test.txt
