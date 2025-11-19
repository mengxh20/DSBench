#!/bin/bash

number=$1
# 定义 JSON 文件所在的目录
JSON_DIR="/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/model_inference/final_inferenced"  # 替换为实际目录路径

# 遍历目录下的所有 .json 文件
for file in "$JSON_DIR"/*.json
do
    # 提取文件名（不带路径和后缀）
    modelname=$(basename "$file" .json)

    # 执行 Python 脚本并传入参数
    echo "Running: python metrics.py --modelname $modelname"
    python metrics.py --modelname "$modelname" --promptnumber ${number}
done