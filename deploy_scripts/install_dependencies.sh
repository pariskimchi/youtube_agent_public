#!/bin/bash


# Install required packages
# sudo apt-get update
# sudo apt-get install -y pkg-config python-venv

# Create and activate virtual environment
# 가상환경 경로
venv_dir="/home/app/myenv"

# 가상환경이 없으면 생성
if [ ! -d "$venv_dir" ]; then
    python3 -m venv $venv_dir
fi

# 가상환경 활성화
source $venv_dir/bin/activate

# 패키지가 requirements.txt에 있는지 확인하고 설치
# requirements.txt 파일 존재 확인
if [ -f "/home/requirements.txt" ]; then
    # requirements.txt에 명시된 패키지 설치 및 업그레이드
    pip install -U -r /home/requirements.txt

    # requirements.txt에 명시된 패키지 목록 저장
    required_packages=$(cat /home/requirements.txt)
    
    # 가상 환경에서 설치된 패키지 목록 저장
    installed_packages=$(pip list)

    # requirements.txt에 명시된 패키지가 가상 환경에 설치되어 있는지 확인하고 설치되어 있지 않은 패키지만 설치
    while read -r package; do
        if ! echo "$installed_packages" | grep -q "$package"; then
            pip install "$package"
        fi
    done <<< "$required_packages"
fi

# 가상환경이 활성화되어 있는지 확인하고 활성화되어 있으면 비활성화
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi