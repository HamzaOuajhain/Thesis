#!/bin/bash

# Path to the directory containing the configuration files
config_dir="$HOME/Desktop/Thesis/Codes/Thesis/Co-SLAM/configs/replica_experiments"

# Path to the Co-SLAM script
coslam_script="$HOME/Desktop/Thesis/Codes/Thesis/Co-SLAM/coslam.py"

# Ensure the base_config_replica.yaml is in the correct location
if [ ! -f "${config_dir}/base_config_replica.yaml" ]; then
    echo "Error: base_config_replica.yaml not found in ${config_dir}"
    exit 1
fi

# Loop through each experiment configuration file
for config_file in ${config_dir}/config_*.yaml
do
    experiment_number=$(basename ${config_file} .yaml | sed 's/config_//')
    echo "Running experiment ${experiment_number} with configuration ${config_file}"

    # Run the Co-SLAM script with the current configuration file
    python ${coslam_script} --config ${config_file}

    # Check if the previous command was successful
    if [ $? -eq 0 ]; then
        echo "Experiment ${experiment_number} completed successfully."
    else
        echo "Experiment ${experiment_number} failed."
    fi

    echo "-----------------------------------"
done

echo "All experiments completed."
