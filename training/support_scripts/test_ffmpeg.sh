#!/bin/bash
for img in $(ls /cvmfs/singularity.metacentrum.cz/NGC/ | grep -i "PyTorch"); do
    FULL_PATH="/cvmfs/singularity.metacentrum.cz/NGC/$img"
    echo "Testing $FULL_PATH"
    singularity exec --nv $FULL_PATH python -c "import cv2; print('$FULL_PATH:', cv2.getBuildInformation())" | grep -i FFMPEG
done
