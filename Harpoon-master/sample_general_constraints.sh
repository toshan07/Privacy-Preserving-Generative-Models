#!/bin/bash

# Define the options for the synth_mask parameter
options_dataset=("adult" "default" "shoppers") # adult shoppers
options_constraint=("range" "category" "both" "or")  #"range" "category"

for dataset in "${options_dataset[@]}"
do
  for constraint in "${options_constraint[@]}"
  do
    python3.12 sampling_GReaT_generalconstraints.py --dataname $dataset --constraint $constraint
    python3.12 sampling_harpoon_ohe_tubular_generalconstraints.py --dataname $dataset --constraint $constraint
    python3.12 sampling_repaint_generalconstraints.py --dataname $dataset --constraint $constraint
#     python3.12 sampling_gain_generalconstraints.py --dataname $dataset --constraint $constraint
  done
done
