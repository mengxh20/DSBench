import asyncio
import json
import re
from collections import defaultdict  # (新) 导入 defaultdict 以便轻松分组
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="指定模型名称和模型提供者")

parser.add_argument("--modelname","-m", type=str, required=True, help="模型名称，例如：gemini-2.5-pro, grok-3, etc.")
parser.add_argument("--promptnumber","-p", type=int, required=True, help="模型名称，例如：gemini-2.5-pro, grok-3, etc.")
# 解析命令行参数
args = parser.parse_args()

# 获取参数值
MODEL_NAME = args.modelname
pro_number = args.promptnumber
# --- 配置区域 ---
API_KEY = "sk-xxx"
API_BASE_URL = "xxx"
# MODEL_NAME = "Qwen2.5-VL-7B-Instruct"  # 注意：这个变量定义了，但在API调用中未使用
INPUT_FILE_PATH = f"/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/model_inference/final_inferenced/{MODEL_NAME}.json"
out_dir = f"/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/model_inference/metrics_result{pro_number}" # resultX代表使用第X个prompt
PROMPT_FILE_PATH = f"/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/mxh_codes/prompt{pro_number}.txt"
os.makedirs(out_dir, exist_ok=True)
output_file_path = f"{out_dir}/{MODEL_NAME}.txt"
# 检查文件是否已存在
if os.path.exists(output_file_path):
    print(f"文件已存在：{output_file_path}. 程序退出。")
    sys.exit(0)  # 退出程序，状态码为 0 表示正常退出


# 设置并发请求的数量，可根据服务器能力调整
CONCURRENT_REQUESTS = 32
# 设置请求重试次数
RETRY_ATTEMPTS = 5

# --- 异步 OpenAI 客户端初始化 ---
client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={"X-Model-Provider-Id": "azure_openai"}
)

# --- 异步任务函数 ---
# (改) 函数现在返回分数和子类型的元组
async def get_score_for_item(item, prompt_template, semaphore):
    """
    异步获取单个数据项的分数，并返回分数及其子类型。

    Args:
        item (dict): 包含问题、答案等信息的数据字典。
        prompt_template (str): 预先加载的提示词模板。
        semaphore (asyncio.Semaphore): 用于控制并发量的信号量。

    Returns:
        tuple[int | None, str]: 包含 (分数, 子类型) 的元组。失败时分数为 None。
    """
    sub_type = item.get('sub_type', 'Unknown')  # 安全地获取sub_type，若不存在则为'Unknown'
    async with semaphore:
        question_text = item['question']
        reference_text = item['ground_truth']
        generated_text = item['predicted_answer']

        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o",  # 温馨提示：这里硬编码了 gpt-4o，而非使用 MODEL_NAME 变量
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_template},
                            {"type": "text", "text": f"The question is {question_text}"},
                            {"type": "text", "text": f"The groundtruth_answer is {reference_text}"},
                            {"type": "text", "text": f"The predicted_answer is {generated_text}"},
                        ],
                    }],
                    stream=False,
                )

                content = response.choices[0].message.content.strip()
                match = re.search(r"[Ff]inal [Ss]core.*?[:\-=\s]*['\"\(\[]*(\d+)[\]\"'\)]*", content)

                if match:
                    print(f"[OUTPUT]:\n {content}")
                    return (int(match.group(1)), sub_type)  # (改) 返回分数和子类型
                else:
                    print(f"[Warning] Could not parse score from response: {content}")
                    return (None, sub_type)  # (改) 解析失败也返回，分数部分为None

            except Exception as e:
                print(f"[Error] Item with question '{question_text[:30]}...' failed on attempt {attempt+1}: {e}")
                if attempt < RETRY_ATTEMPTS - 1:
                    await asyncio.sleep(2)
                else:
                    print(f"[Error] Item failed after {RETRY_ATTEMPTS} attempts. Skipping.")
                    return (None, sub_type)  # (改) 所有重试失败后也返回

# --- 主执行函数 ---
async def main():
    """
    主执行函数，负责加载数据、创建和运行并发任务，并按子类型分类统计结果。
    """
    try:
        with open(INPUT_FILE_PATH, "r", encoding="utf-8") as file:
            data = json.load(file)
        with open(PROMPT_FILE_PATH, "r", encoding="utf-8") as file:
            prompt = file.read()
    except FileNotFoundError as e:
        print(f"Error loading file: {e}")
        return

    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    tasks = [get_score_for_item(item, prompt, semaphore) for item in data]

    print(f"Starting {len(tasks)} concurrent requests with a concurrency limit of {CONCURRENT_REQUESTS}...")

    # results 将会是 [(score, sub_type), (score, sub_type), ...] 形式的列表
    results = await tqdm.gather(*tasks)

    # --- (新) 结果处理与分类统计 ---
    scores_by_subtype = defaultdict(list)
    successful_requests = 0

    for score, sub_type in results:
        if score is not None:
            scores_by_subtype[sub_type].append(score)
            successful_requests += 1

    total_requests = len(data)
    
    print("\n--- Inference Complete ---")
    print(f"Total items: {total_requests}")
    print(f"Successfully processed: {successful_requests}")
    print(f"Failed items: {total_requests - successful_requests}")

    if not successful_requests:
        print("No scores were successfully calculated.")
        return

    
    with open(output_file_path, "w", encoding="utf-8") as file:
        print("\n--- Score Breakdown by Subtype ---")
        # 按子类型字母顺序排序，使输出更规整
        for sub_type, scores in sorted(scores_by_subtype.items()):
            if scores:
                average_score = sum(scores) / len(scores)
                # print(f"- Subtype: {sub_type}")
                # print(f"  - Average Score: {average_score:.2f}")
                # print(f"  - Item Count: {len(scores)}")
                file.write(f"- Subtype: {sub_type}\n")
                file.write(f"  - Average Score: {average_score:.2f}\n")
                file.write(f"  - Item Count: {len(scores)}\n")

        # --- (新) 计算并打印总平均分 ---
        total_score = sum(sum(scores) for scores in scores_by_subtype.values())
        overall_average = total_score / successful_requests if successful_requests > 0 else 0
        # print("\n--- Overall Performance ---")
        # print(f"{MODEL_NAME}'s overall average_score: {overall_average:.2f}")
        file.write("\n--- Overall Performance ---\n")
        file.write(f"{MODEL_NAME}'s overall average_score: {overall_average:.2f}\n")


if __name__ == "__main__":
    asyncio.run(main())