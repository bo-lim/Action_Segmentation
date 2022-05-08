#!/bin/bash
split="1 2 3 4 5"
for sp in $split
do
  python main.py --action=train --split=$sp \
                --num_epochs=100 --dirname sc6 --section_num 6
done