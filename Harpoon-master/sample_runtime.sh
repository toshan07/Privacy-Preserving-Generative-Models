#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("gesture" "magic" "bean" "california" "letter")
options_mask=("MAR")
options_ratio=("0.25")

for dataset in "${options_dataset[@]}"
do
  for mask in "${options_mask[@]}"
  do
    for ratio in "${options_ratio[@]}"
    do
      python3.12 sampling_harpoon_ohe_tubular.py --dataname $dataset --mask $mask --ratio $ratio --loss mae --runtime_test True
      python3.12 sampling_GReaT.py --dataname $dataset --mask $mask --ratio $ratio --runtime_test True
      python3.12 sampling_repaint.py --dataname $dataset --mask $mask --ratio $ratio --runtime_test True
      python3.12 sampling_remasker.py --dataname $dataset --mask $mask --ratio $ratio --runtime_test True
      python3.12 sampling_gain.py --dataname $dataset --mask $mask --ratio $ratio --runtime_test True
      python3.12 sampling_miracle.py --dataname $dataset --mask $mask --ratio $ratio --runtime_test True
    done
  done
done