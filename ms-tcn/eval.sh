#!/bin/bash
split="1 2 3 4 5"

for sp in $split
do
  echo "split : "$sp
  python eval.py --version 2 --split $sp --num_epoch ${1}
done