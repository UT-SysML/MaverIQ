#!/bin/bash

# Data for Figure 10, 12, 13
model_list_values=("large")
scale_factors=(0.1 0.2 0.3 0.4)
trace_values=("code conversation")
mapping_policy_values=("balanced")
quant_values=("16bit")
modes=("trace" "memory")

for model_list in "${model_list_values[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    for trace in "${trace_values[@]}"; do
      for mapping_policy in "${mapping_policy_values[@]}"; do
        for quant in "${quant_values[@]}"; do
          for mode in "${modes[@]}"; do
            python3 accelerate_trace_replay.py --scale_factor "$scale_factor" --trace "$trace" --model_list "$model_list" --mapping_policy "$mapping_policy" --quant "$quant" --mode "$mode"
          done
        done
      done
    done
  done
done


# Data for Figure 11
model_list_values=("regular")
scale_factors=(0.4)
trace_values=("code")
mapping_policy_values=("balanced")
quant_values=("16bit")
modes=("trace" "memory")

for model_list in "${model_list_values[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    for trace in "${trace_values[@]}"; do
      for mapping_policy in "${mapping_policy_values[@]}"; do
        for quant in "${quant_values[@]}"; do
          for mode in "${modes[@]}"; do
            python3 accelerate_trace_replay.py --scale_factor "$scale_factor" --trace "$trace" --model_list "$model_list" --mapping_policy "$mapping_policy" --quant "$quant" --mode "$mode"
          done
        done
      done
    done
  done
done


# Data for Figure 14
model_list_values=("twentyone")
scale_factors=(0.2 0.4)
trace_values=("conversation")
mapping_policy_values=("balanced")
quant_values=("16bit")
modes=("trace" "memory")

for model_list in "${model_list_values[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    for trace in "${trace_values[@]}"; do
      for mapping_policy in "${mapping_policy_values[@]}"; do
        for quant in "${quant_values[@]}"; do
          for mode in "${modes[@]}"; do
            python3 accelerate_trace_replay.py --scale_factor "$scale_factor" --trace "$trace" --model_list "$model_list" --mapping_policy "$mapping_policy" --quant "$quant" --mode "$mode"
          done
        done
      done
    done
  done
done


# Extra Data for SLO Experiment 
model_list_values=("large")
scale_factors=(0.15 0.25)
trace_values=("conversation")
mapping_policy_values=("balanced")
quant_values=("16bit")
modes=("trace" "memory")

for model_list in "${model_list_values[@]}"; do
  for scale_factor in "${scale_factors[@]}"; do
    for trace in "${trace_values[@]}"; do
      for mapping_policy in "${mapping_policy_values[@]}"; do
        for quant in "${quant_values[@]}"; do
          for mode in "${modes[@]}"; do
            python3 accelerate_trace_replay.py --scale_factor "$scale_factor" --trace "$trace" --model_list "$model_list" --mapping_policy "$mapping_policy" --quant "$quant" --mode "$mode"
          done
        done
      done
    done
  done
done