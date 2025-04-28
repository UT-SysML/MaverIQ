#!/bin/bash

########################################################################################################################
############################################## Data for Figures 10, 12, 13 #############################################
########################################################################################################################
# AlpaServe
# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.2 0.3 0.4)
trace_sels=(code conversation)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."

# AlpaServe*
# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.2 0.3 0.4)
trace_sels=(code conversation)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe*" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."

########################################################################################################################
################################################## Data for Figure 11 ##################################################
########################################################################################################################
# AlpaServe
# Define arrays for rate_scale and trace_sel
rate_scales=(0.4)
trace_sels=(code)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe" \
                --model_list regular \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe_regular_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe_regular_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."

# AlpaServe*
# Define arrays for rate_scale and trace_sel
rate_scales=(0.4)
trace_sels=(code)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe*" \
                --model_list regular \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe*_regular_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe*_regular_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."


########################################################################################################################
################################################## Data for Figure 14 ##################################################
########################################################################################################################
# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator_baselines.py with specified arguments: twentyone - 0.4 - conversation - 5
python3 -u MS_trace_generator_baselines.py --baseline AlpaServe --model_list twentyone --scale_factor 0.4 --trace conversation --slo 5.0  > trace_AlpaServe_twentyone_None_0.4_None_conversation_0.5.txt

# Kill mpirun and python3 processes and wait 10 seconds to ensure processes are terminated
pkill -9 python3
pkill -9 mpirun
sleep 10
# Run MS_trace_generator_baselines.py with specified arguments: twentyone - 0.2 - conversation - 5
python3 -u MS_trace_generator_baselines.py --baseline AlpaServe --model_list twentyone --scale_factor 0.2 --trace conversation --slo 5.0  > trace_AlpaServe_twentyone_None_0.2_None_conversation_0.5.txt


########################################################################################################################
############################################# Extra Data for SLO Experiment ############################################
########################################################################################################################
# AlpaServe
# Define arrays for rate_scale and trace_sel
rate_scales=(0.15 0.25)
trace_sels=(conversation)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."

# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.15 0.2 0.25 0.3 0.4)
trace_sels=(conversation)
slo_sels=(2.5 7.5)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."


# AlpaServe*
# Define arrays for rate_scale and trace_sel
rate_scales=(0.15 0.25)
trace_sels=(conversation)
slo_sels=(5.0)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe*" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."

# Define arrays for rate_scale and trace_sel
rate_scales=(0.1 0.15 0.2 0.25 0.3 0.4)
trace_sels=(conversation)
slo_sels=(2.5 7.5)

# Nested loop over rate_scale and trace_sel
for rate_scale in "${rate_scales[@]}"; do
    for trace_sel in "${trace_sels[@]}"; do
        for slo_sel in "${slo_sels[@]}"; do
            echo "Processing rate_scale=$rate_scale, trace_sel=$trace_sel, slo_sel=$slo_sel..."

            # Kill mpirun and python3 processes
            echo "Killing mpirun and python3 processes..."
            pkill -9 mpirun
            pkill -9 python3

            # Wait 10 seconds to ensure processes are terminated
            echo "Waiting 10 seconds..."
            sleep 10

            # Run MS_trace_generator.py with specified arguments
            echo "Running MS_trace_generator_baselines.py..."
            python3 -u MS_trace_generator_baselines.py \
                --baseline "AlpaServe*" \
                --model_list large \
                --scale_factor "$rate_scale" \
                --trace "$trace_sel" \
                --slo "$slo_sel" \
                > "trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"

            echo "Completed: Output saved to trace_AlpaServe*_large_None_${rate_scale}_None_${trace_sel}_${slo_sel}.txt"
        done
    done
done

echo "Script finished."