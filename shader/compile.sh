#!/bin/bash

# 프로젝트의 특정 디렉토리 위치 보장을 위해 현재 스크립트 위치로 이동
cd "$(dirname "$0")"

# 컴파일된 바이너리가 들어갈 spv 폴더가 없다면 생성
mkdir -p spv

# glsl 폴더 안의 모든 .comp 파일을 컴파일
for shader in glsl/*.comp; do
    filename=$(basename "$shader")
    name="${filename%.*}"
    
    echo "Compiling $filename -> spv/$name.spv"
    
    # glslc 명령어를 이용해 컴파일. 최적화 플래그(-O) 및 subgroup 연산을 위한 Vulkan 1.2 타겟 추가
    glslc "$shader" -o "spv/$name.spv" -O --target-env=vulkan1.2
    
    # 컴파일 에러 발생 시 중단
    if [ $? -ne 0 ]; then
        echo "Error compiling $filename"
        exit 1
    fi
done

echo "All shaders compiled successfully!"
