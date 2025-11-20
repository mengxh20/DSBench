import os
import json
import base64
import asyncio
import time
import random
import signal
import sys
from tqdm import tqdm
import aiohttp
import argparse
# =========================================================
# 工具函数
# =========================================================
def encode_image(path: str) -> str:
    """将图像文件转为 base64 字符串"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# =========================================================
# 基础配置
# =========================================================
API_KEY = "sk-xxx"
API_BASE_URL = "xxx"

# 异步初始化
session = None

# 输入文件路径
input_file = "/dataset_path/DSBench_QAs.json"

# =========================================================
# 自动生成输出文件名
# =========================================================
# model_name_lst = ['grok-3', 'qwen-vl-max' ,'Qwen/Qwen2.5-VL-32B-Instruct','Qwen/Qwen2.5-VL-72B-Instruct','moonshotai/Kimi-K2-Instruct','doubao-seed-1-6-250615' ]
# XModelProviderId_lst = ['azure_openai', 'tongyi', 'siliconflow','siliconflow','siliconflow','volcengine_maas']
# model_name = "gemini-2.5-pro"   #  grok-3        qwen-vl-max    Qwen/Qwen2.5-VL-32B-Instruct  
# XModelProviderId = "vertex_ai"  #  azure_openai  tongyi
# 提取任务名（根据文件路径）
task_name = os.path.basename(os.path.dirname(input_file))

# 自动创建输出目录
output_dir = "./model_inference"
os.makedirs(output_dir, exist_ok=True)

# # 拼接输出文件名
# output_file = os.path.join(output_dir, f"{model_name}.json")

# print(f"[INFO] 输出文件将保存为: {output_file}")


# 新增：控制并发数的 Semaphore
MAX_CONCURRENT_REQUESTS = 16  # 设置最大并发请求数，根据实际需求调整
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
async def run_inference(session, image_path: str, question: str, model_name:str, XModelProviderId:str):
    """异步调用模型执行单条推理（带并发限制）"""
    async with semaphore:  # 使用信号量控制并发数
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
            "X-Model-Provider-Id": XModelProviderId
        }

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(image_path)}"}}
                    ]
                }
            ]
        }

        try:
            async with session.post(API_BASE_URL + "/chat/completions", json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["choices"][0]["message"]["content"]
                else:
                    return f"ERROR: {response.status} - {await response.text()}"
        except Exception as e:
            return f"ERROR: {str(e)}"


# =========================================================
# 主逻辑：读取全部数据，处理并推理
# =========================================================
async def main():
    global session
    session = aiohttp.ClientSession()
    parser = argparse.ArgumentParser(description="指定模型名称和模型提供者")

    # 定义命令行参数
    parser.add_argument("--modelname", type=str, required=True, help="模型名称，例如：gemini-2.5-pro, grok-3, etc.")
    parser.add_argument("--Xmodel", type=str, required=True, help="模型提供者 ID，例如：vertex_ai, Qwen/Qwen2.5-VL-Instruct, etc.")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取参数值
    model_name = args.modelname
    XModelProviderId = args.Xmodel
    # 拼接输出文件名
    output_file = os.path.join(output_dir, f"{model_name}.json")

    print(f"[INFO] 输出文件将保存为: {output_file}")
    # 加载输入数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"共读取 {total_samples} 条数据。")

    # 加载已有结果（断点续推）
    results = []
    processed_images = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
                processed_images = {r["image_path"] for r in results}
                print(f"已加载 {len(results)} 条历史结果，将跳过这些样本。")
            except Exception:
                results = []
                processed_images = set()

    # 安全退出函数（中断时保存）
    def handle_interrupt(signum, frame):
        print("\n检测到中断信号，正在保存中间结果...")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"已保存 {len(results)} 条推理结果到 {output_file}")
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_interrupt)
    signal.signal(signal.SIGTERM, handle_interrupt)

    # 遍历并推理
    tasks = []
    for item in tqdm(data, desc="Running inference"):
        try:
            image_path = item["image_path"]
            question = item["question"]
            ground_truth = item["ground_truth"]

            if image_path in processed_images:
                continue  # 跳过已处理样本

            # 异步处理每个推理请求
            task = asyncio.create_task(run_inference(session, image_path, question,model_name,XModelProviderId))
            tasks.append((task, item, ground_truth, image_path))

        except Exception as e:
            print(f"[ERROR] 样本处理失败: {e}")
            result_item = {
                "image_path": item.get("image_path", "unknown"),
                "question": item["question"],
                "predicted_answer": f"ERROR: {str(e)}",
                "ground_truth": item["ground_truth"]
            }
            results.append(result_item)
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

    # 等待所有异步任务完成并处理结果
    for task, item, ground_truth, image_path in tqdm(tasks, desc="Processing results"):
        predicted_answer = await task

        result_item = {
            "image_path": image_path,
            "question": item["question"],
            "predicted_answer": predicted_answer,
            "ground_truth": ground_truth
        }

        results.append(result_item)
        processed_images.add(image_path)

        # 实时保存结果（防崩溃丢失）
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    print(f"\n推理完成，共处理 {len(results)} 条样本。")
    print(f"结果已保存至: {output_file}")

    # 关闭 session
    await session.close()

# 运行程序
if __name__ == "__main__":
    asyncio.run(main())