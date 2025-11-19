import json
import os

def convert_to_multimodal_format(input_file_path, output_file_path):
    """
    将特定格式的JSON数据转换为多模态对话格式。

    Args:
        input_file_path (str): 输入的JSON文件路径。
        output_file_path (str): 转换后要保存的JSON文件路径。
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误：输入文件 '{input_file_path}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：输入文件 '{input_file_path}' 不是有效的JSON格式。")
        return

    converted_data = []

    for item in original_data:
        
        if 'image_path' not in item or 'question' not in item or 'ground_truth' not in item:
            print(f"警告：跳过一条不完整的记录：{item}")
            continue

       
        image_path = item['image_path']
        question = item['question']
        ground_truth = item['ground_truth']

       
        new_item = {
            "conversations": [
                {
                    "from": "human",
                    # 将 <image> 标记放在问题开头
                    "value": f"<image>\n{question}"
                },
                {
                    "from": "gpt",
                    "value": ground_truth
                }
            ],
            "images": [
                image_path
            ]
        }
        converted_data.append(new_item)

    # 确保存储输出文件的目录存在
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

   
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, indent=4, ensure_ascii=False)

    print(f"转换成功！数据已保存到 '{output_file_path}'。")


if __name__ == '__main__':
    # --- 请在这里配置您的文件路径 ---
    
    # 您的原始JSON文件
    input_json_file = '/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/ADqa_output_cabin_rewritten_finalversion_shuffle.json' 
    
    # 您希望保存的新JSON文件的路径
    output_json_file = '/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/ADqa_output_cabin_rewritten_finalversion_shuffle_train.json'
    
    # 运行转换函数
    convert_to_multimodal_format(input_json_file, output_json_file)