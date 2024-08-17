 #!/bin/bash

scenes=(
    "office2-depth"
    "office2-rgb"
    "office2"
)

for scene in "${scenes[@]}"
do
    echo "Running CoSLAM for scene: $scene"
    python coslam.py --config "./configs/Replica/${scene}.yaml"
    echo "Finished processing $scene"
    echo "------------------------"
done

echo "All scenes processed."
