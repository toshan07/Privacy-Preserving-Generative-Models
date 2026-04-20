#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "bean" "california" "default" "gesture" "letter" "magic" "shoppers")


for dataset in "${options_dataset[@]}"
do
#  python3.12 train_tabddpm.py --dataname $dataset
#  python3.12 train_harpoon_ordinal.py --dataname $dataset
#  python3.12 train_gain.py --dataname $dataset
#  python3.12 train_remasker.py --dataname $dataset
#  python3.12 train_GReaT.py --dataname $dataset
#  python3.12 train_repaint.py --dataname $dataset
#  python3.12 train_diffputer.py --dataname $dataset
   python3.12 train_repaint_quantile.py --dataname $dataset
done
