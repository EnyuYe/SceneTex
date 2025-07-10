#!/bin/bash  
# 设置GPU数量和环境变量  
export CUDA_VISIBLE_DEVICES=0,1,2,3  
export MASTER_ADDR=localhost  
export MASTER_PORT=12355  
  
# 获取参数  
stamp=${1:-$(date "+%Y-%m-%d_%H-%M-%S")}  
log_dir=${2:-"outputs/"}  
prompt=${3:-"a bohemian style living room"}  
scene_id=${4:-"93f59740-4b65-4e8b-8a0f-6420b339469d/room_4"}  
  
# 启动多GPU训练  
torchrun --nproc_per_node=4 scripts/train_texture.py \  
    --config config/template.yaml \  
    --stamp $stamp \  
    --log_dir $log_dir \  
    --prompt "$prompt" \  
    --scene_id "$scene_id"