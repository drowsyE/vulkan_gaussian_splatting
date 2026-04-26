#!/bin/bash

cd "$(dirname "$0")"

mkdir -p spv

for shader in glsl/*.comp; do
    filename=$(basename "$shader")
    name="${filename%.*}"
    
    echo "Compiling $filename -> spv/$name.spv"
    
    glslc "$shader" -o "spv/$name.spv" -O --target-env=vulkan1.2
    
    if [ $? -ne 0 ]; then
        echo "Error compiling $filename"
        exit 1
    fi
done

echo "All shaders compiled successfully!"
