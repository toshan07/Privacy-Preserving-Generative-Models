#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "bean" "california" "default" "gesture" "letter" "magic" "shoppers")
options_mask=("MCAR")
options_ratio=("0.00")

for dataset in "${options_dataset[@]}"
do
  for mask in "${options_mask[@]}"
  do
    for ratio in "${options_ratio[@]}"
    do
      python3.12 sampling_harpoon_ohe_tubular.py --dataname $dataset --mask $mask --ratio $ratio --loss mae --ignore_hard_fix True
#      python3.12 sampling_harpoon_ohe_tubular.py --dataname $dataset --mask $mask --ratio $ratio --loss mse_kld
#      python3.12 sampling_harpoon_ohe_basicmanifold_kld.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_harpoon_ordinal.py --dataname $dataset --mask $mask --ratio $ratio --loss mae
#      python3.12 sampling_harpoon_ordinal.py --dataname $dataset --mask $mask --ratio $ratio --loss mse
#      python3.12 sampling_harpoon_ohe_basicmanifold.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_GReaT.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_repaint.py --dataname $dataset --mask $mask --ratio $ratio --ignore_hard_fix True
#      python3.12 sampling_diffputer.py --dataname $dataset --mask $mask --ratio $ratio --num_steps 50 --num_trials 5
#      python3.12 sampling_remasker.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_gain.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_hyperimpute.py --dataname $dataset --mask $mask --ratio $ratio
#      python3.12 sampling_miracle.py --dataname $dataset --mask $mask --ratio $ratio
    done
  done
done
