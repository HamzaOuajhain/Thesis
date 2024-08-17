#!/bin/bash

# Change to the directory containing coslam.py
cd ~/Desktop/Thesis/Codes/Thesis/Co-SLAM

# Directory containing the YAML files
config_dir="configs/Box_experiments"

# Log file
log_file="box_experiments_log.txt"

# Clear the log file
> $log_file

# Iterate through all YAML files in the Box_experiments directory
for config in $config_dir/*.yaml
do
    if [ "$config" != "$config_dir/Base_box.yaml" ]; then
        echo "Running experiment with config: $config"
        echo "Running experiment with config: $config" >> $log_file
        
        # Run coslam.py with the current config file
        python coslam.py --config "$config" 2>&1 | tee -a $log_file
        
        # Check if the command was successful
        if [ $? -eq 0 ]; then
            echo "Experiment completed successfully" >> $log_file
        else
            echo "Error occurred during experiment" >> $log_file
        fi
        
        echo "----------------------------------------" >> $log_file
    fi
done

echo "All experiments completed. Check $log_file for details."