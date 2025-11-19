import re
import os
import sys
from collections import defaultdict
from openai import OpenAI

# --- 配置部分 ---
API_KEY = "sk-xxx"
API_BASE_URL = "xxx"

# 文件路径和输出路径
INPUT_FILE_PATH = "/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/mxh_codes/out_txt.txt"  # 输入的 txt 文件路径
OUTPUT_FILE_PATH = "/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/mxh_codes/scores_output.txt"  # 输出评分结果的文件路径

# --- 同步 OpenAI 客户端初始化 ---
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={"X-Model-Provider-Id": "azure_openai"}
)

# --- 同步评分函数 ---
def get_score_for_response(question, ground_truth, response, model_name, prompt_template):
    """
    获取单个模型回复的 GPT 评分。

    Args:
        question (str): 问题文本。
        ground_truth (str): Ground Truth 标准答案。
        response (str): 模型的回复内容。
        model_name (str): 模型名称。
        prompt_template (str): 评分提示模板。

    Returns:
        tuple[str, int | None]: 返回模型名称与得分。
    """
    for attempt in range(3):  # 重试次数
        try:
            # 发送请求给 GPT
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt_template},
                    {"role": "user", "content": f"The question is: {question}"},
                    {"role": "user", "content": f"The groundtruth answer is: {ground_truth}"},
                    {"role": "user", "content": f"The model's answer is: {response}"}
                ],
                stream=False,
            )

            # 提取 GPT 返回的分数
            content = response.choices[0].message.content.strip()
            match = re.search(r"[Ff]inal [Ss]core.*?[:\-=\s]*['\"(\[]*(\d+)[\]\"')]*", content)
            if match:
                return model_name, int(match.group(1))  # 返回分数
            else:
                print(f"[Warning] Failed to parse response for model {model_name}: {content}")
                return model_name, None
        except Exception as e:
            print(f"[Error] Failed to process model {model_name} on attempt {attempt+1}: {e}")
            if attempt == 2:  # 如果是最后一次尝试，返回 None
                return model_name, None

# --- 主函数 ---
def main():
    """
    主函数，负责读取文本文件、解析数据、并调用 GPT 对结果评分。
    """
    # 检查输出文件是否已存在
    if os.path.exists(OUTPUT_FILE_PATH):
        print(f"Output file already exists: {OUTPUT_FILE_PATH}. Exiting.")
        sys.exit(0)

    # 读取输入文件内容
    try:
        with open(INPUT_FILE_PATH, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Input file not found: {INPUT_FILE_PATH}")
        return

    # 解析文件中的 Q、GT 和各模型的回复
    questions = []
    ground_truths = []
    responses = defaultdict(list)  # 按模型名称存储回复

    current_question = None
    current_gt = None

    for line in lines:
        if line.startswith("Q:"):
            current_question = line[2:].strip()
        elif line.startswith("GT:"):
            current_gt = line[3:].strip()
        elif any(line.startswith(model) for model in ["Qwen2.5-VL-7B", "Qwen2.5-VL-32B", "Qwen3vlplus", "Ours"]):
            model_name, response = line.split(maxsplit=1)
            responses[model_name.strip()].append((current_question, current_gt, response.strip()))

    # 加载用于评分的 GPT 提示模板
    prompt_template = """
    As an autonomous driving evaluation assistant, your task is to score the relevance and correctness of the provided model's answer. Scores range from 0-100:
    - 0: Completely irrelevant answer
    - 50: Partially correct but inaccurate answer
    - 100: Highly relevant and correct answer
    Provide the Final Score as the result.
    """

    # 同步处理：为每个模型的每个回复打分
    print("Starting GPT scoring...")
    results = []
    for model_name, response_list in responses.items():
        for question, gt, response in response_list:
            result = get_score_for_response(question, gt, response, model_name, prompt_template)
            results.append(result)

    # 整理结果并写入输出文件
    scores_by_model = defaultdict(list)
    for model_name, score in results:
        if score is not None:
            scores_by_model[model_name].append(score)

    with open(OUTPUT_FILE_PATH, "w", encoding="utf-8") as file:
        print("\n--- Model Scores ---")
        for model, scores in scores_by_model.items():
            avg_score = sum(scores) / len(scores) if scores else 0
            file.write(f"Model: {model}\n")
            file.write(f"  - Average Score: {avg_score:.2f}\n")
            file.write(f"  - Item Count: {len(scores)}\n")

    print(f"Scores have been saved to {OUTPUT_FILE_PATH}")

# --- 启动程序 ---
if __name__ == "__main__":
    main()