#!/bin/bash

########################################################################################################################
################################################## Data for Figure 10 ##################################################
########################################################################################################################
# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.2 0.3 0.4)
trace_sels=(code conversation)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel..."

        # Kill mpirun and python3 processes
        echo "Killing mpirun and python3 processes..."
        pkill -9 mpirun
        pkill -9 python3

        # Wait 10 seconds to ensure processes are terminated
        echo "Waiting 10 seconds..."
        sleep 10

        # Run MS_trace_generator.py with specified arguments
        echo "Running MS_trace_generator.py..."
        python3 -u MS_trace_generator.py \
            --model_list large \
            --load_strategy hybrid \
            --packing_threshold 0.7 \
            --scale_factor "$rate_scale" \
            --cost_func min_lat \
            --trace "$trace_sel" \
            --slo 0 \
            > trace_MaverIQ_large_hybrid_0.7_${rate_scale}_min_lat_${trace_sel}_0.txt

        echo "Completed: Output saved to trace_MaverIQ_large_hybrid_0.7_${rate_scale}_min_lat_${trace_sel}_0.txt"
    done
done

echo "Script finished."


########################################################################################################################
################################################## Data for Figure 11 ##################################################
########################################################################################################################
# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10

# Run MS_trace_generator.py with specified arguments: regular - hybrid (0.7) - 0.4 - min_lat - code - 0
python3 -u MS_trace_generator.py --model_list regular --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_lat --trace code --slo 0 > trace_MaverIQ_regular_hybrid_0.7_0.4_min_lat_code_0.txt


########################################################################################################################
################################################## Data for Figure 12 ##################################################
########################################################################################################################
# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.2 - min_cost - code - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.2 --cost_func min_cost --trace code --slo 0 --use_float16_only > trace_MaverIQ-FP16_large_hybrid_0.7_0.2_min_cost_code_0.txt

# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.4 - min_cost - code - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_cost --trace code --slo 0 --use_float16_only > trace_MaverIQ-FP16_large_hybrid_0.7_0.4_min_cost_code_0.txt

# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.4 - min_cost - conversation - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_cost --trace conversation --slo 0 --use_float16_only > trace_MaverIQ-FP16_large_hybrid_0.7_0.4_min_cost_conversation_0.txt


########################################################################################################################
################################################## Data for Figure 13 ##################################################
########################################################################################################################
# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.4 - min_cost - code - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_cost --trace code --slo 0 > trace_MaverIQ_large_hybrid_0.7_0.4_min_cost_code_0.txt

# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.4 - min_mem - code - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_mem --trace code --slo 0 > trace_MaverIQ_large_hybrid_0.7_0.4_min_mem_code_0.txt

# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator.py with specified arguments: large - hybrid (0.7) - 0.4 - min_gpu_cost - code - 0
python3 -u MS_trace_generator.py --model_list large --load_strategy hybrid --packing_threshold 0.7 --scale_factor 0.4 --cost_func min_gpu_cost --trace code --slo 0 > trace_MaverIQ_large_hybrid_0.7_0.4_min_gpu_cost_code_0.txt


########################################################################################################################
################################################## Data for Figure 14 ##################################################
########################################################################################################################
modes=(min_lat min_cost min_mem min_gpu_cost)
rate_scales=(0.2 0.4)

for mode in "${modes[@]}"; do
    for rate_scale in "${rate_scales[@]}"; do
        echo "Processing mode=$mode with rate_scale=$rate_scale..."

        # Kill mpirun and python3 processes
        echo "Killing mpirun and python3 processes..."
        pkill -9 mpirun
        pkill -9 python3

        # Wait 10 seconds to ensure processes are terminated
        echo "Waiting 10 seconds..."
        sleep 10

        # Run MS_trace_generator.py with specified arguments
        echo "Running MS_trace_generator.py..."
        python3 -u MS_trace_generator.py \
            --model_list twentyone \
            --load_strategy hybrid \
            --packing_threshold 0.7 \
            --scale_factor "$rate_scale" \
            --cost_func "$mode" \
            --trace conversation \
            --slo 0 \
            > trace_MaverIQ_twentyone_hybrid_0.7_${rate_scale}_${mode}_conversation_0.txt

        echo "Completed: Output saved to trace_MaverIQ_twentyone_hybrid_0.7_${rate_scale}_${mode}_conversation_0.txt"
    done
done

echo "Script finished."


########################################################################################################################
################################################ Data for SLO Experiment ###############################################
########################################################################################################################
# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.15 0.2 0.25 0.3 0.4)
slo_sels=(2.5 5.0 7.5)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for slo_sel in "${slo_sels[@]}"; do
        echo "Processing rate_scale=$rate_scale, slo_sel=$slo_sel..."

        # Kill mpirun and python3 processes
        echo "Killing mpirun and python3 processes..."
        pkill -9 mpirun
        pkill -9 python3

        # Wait 10 seconds to ensure processes are terminated
        echo "Waiting 10 seconds..."
        sleep 10

        # Run MS_trace_generator.py with specified arguments
        echo "Running MS_trace_generator.py..."
        python3 -u MS_trace_generator.py \
            --model_list large \
            --load_strategy hybrid \
            --packing_threshold 0.7 \
            --scale_factor "$rate_scale" \
            --cost_func min_cost \
            --trace conversation \
            --slo "$slo_sel" \
            > trace_MaverIQ_large_hybrid_0.7_${rate_scale}_min_cost_conversation_${slo_sel}.txt

        echo "Completed: Output saved to trace_MaverIQ_large_hybrid_0.7_${rate_scale}_min_cost_conversation_${slo_sel}.txt"
    done
done

echo "Script finished."