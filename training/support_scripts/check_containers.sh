#!/bin/bash

echo "🔍 Kontrola PyTorch kontejnerů: Python >= 3.7, cv2, FFMPEG a torch verze"
echo

# Pomocná funkce pro kontrolu verze Pythonu
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
    echo "🧪 $img"

    # Verze Pythonu
    version_output=$(singularity exec $FULL_PATH python --version 2>&1)
    version=$(echo "$version_output" | grep -oP 'Python \K[0-9]+\.[0-9]+')

    if [ -z "$version" ]; then
        echo "   ❌ Python nenalezen"
        continue
    fi

    echo "   🐍 Python: $version"

    if ! is_valid_python "$version"; then
        echo "   ❌ Verze Pythonu je příliš stará (<3.7)"
        continue
    fi

    # Kontrola modulu cv2
    singularity exec $FULL_PATH python -c "import cv2" &>/dev/null
    if [ $? -ne 0 ]; then
        echo "   ❌ Modul cv2 chybí"
        continue
    fi
    echo "   ✅ Modul cv2 přítomen"

    # FFMPEG podpora
    ffmpeg_support=$(singularity exec $FULL_PATH python -c "import cv2; print(cv2.getBuildInformation())" | grep -i "FFMPEG")

    if [[ "$ffmpeg_support" == *"YES"* ]]; then
        echo "   🎞️ FFMPEG podpora: ✅"
    else
        echo "   🎞️ FFMPEG podpora: ❌"
    fi

    # Verze PyTorch
    torch_version=$(singularity exec $FULL_PATH python -c "import torch; print(torch.__version__)" 2>/dev/null)

    if [ -z "$torch_version" ]; then
        echo "   ❌ torch není dostupný"
    else
        echo "   🔥 torch verze: $torch_version"
    fi

    echo
done
