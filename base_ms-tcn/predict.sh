#!/bin/bash
epoch="10 20 30 40 50 60 70 80 90 92 94 96 98 100"
split="1 2 3 4 5"

for sp in $split
do
  for ep in $epoch
  do
    python main.py --version 2 --action predict --split $sp --num_epochs $ep --dirname sc6
  done
done