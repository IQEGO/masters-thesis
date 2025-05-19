#!/bin/bash

echo "Checking PyTorch containers with Python >= 3.7 support..."
echo

# Regex pro kontrolu Python >= 3.7
is_valid_python() {
    version=$1
    major=$(echo $version | cut -d. -f1)
    minor=$(echo $version | cut -d. -f2)
    if [ "$major" -gt 3 ]; then
        return 0
    elif [ "$major" -eq 3 ] && [ "$minor" -ge 7 ]; then
        return 0
    else
        return 1
    fi
}

for img in $(ls /cvmfs/singularity.metacentrum.cz/NGC/ | grep -i "PyTorch"); do
    FULL_PATH="/cvmfs/singularity.metacentrum.cz/NGC/$img"
    echo -n "$img -> Python version: "
    
    # Získání verze Pythonu
    version_output=$(singularity exec $FULL_PATH python --version 2>&1)
    version=$(echo "$version_output" | grep -oP 'Python \K[0-9]+\.[0-9]+')

    if [ -z "$version" ]; then
        echo "❌ Python not found"
        continue
    fi

    echo -n "$version "

    if is_valid_python "$version"; then
        echo "✅ (OK)"
    else
        echo "❌ (too old)"
    fi
done
