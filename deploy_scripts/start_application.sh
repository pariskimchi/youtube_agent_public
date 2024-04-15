#!/bin/bash 

## 

# virutal environment
venv_dir="/home/app/myenv"

# 가상환경이 없으면 생성
# if [ ! -d "$venv_dir" ]; then
#     # 가상환경 활성화
#     source $venv_dir/bin/activate
# fi

# run virtual env
source $venv_dir/bin/activate

# run streamlit application 
streamlit run /home/app.py