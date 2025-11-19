#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import base64
import random
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# =========================================================
# 基础配置
# =========================================================
API_KEY = "sk-xxx"
API_BASE_URL = "xxx"

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    default_headers={"X-Model-Provider-Id": "azure_openai"}
)

# =========================================================
# 参数配置
# =========================================================
DATASET_DIR = "/e2e-data/evad-tech-vla/mxh24/AD-SAFETY/datasets"
OUTPUT_PATH = "classified_images.json"
STATS_PATH = "dataset_stats.txt"
TARGET_TOTAL = 100_000
MAX_WORKERS = 10
BATCH_SIZE = 50

# =========================================================
# 分类层级定义（仅 external）
# =========================================================
TYPE_HIERARCHY = {
    "external": {
        "Traffic rule": ["Traffic signal", "warning signs"],
        "Lane line": ["Solid line", "Dotted line"],
        "Intersection": ["Crossroads", "Zebra crossing"],
        "Weather category": ["Rainy day", "Foggy day"],
        "Static obstacle": ["Roadblock", "Waterlog"],
        "Dynamic obstacle": [
            "Pedestrians cross",
            "Vehicle slow",
            "Vehicle changing lane"
        ]
    }
}

# =========================================================
# 图像分类函数
# =========================================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def classify_image(image_path, max_retries=3):
    """
    分类函数：模型只需要判断 sub_type 和 sub_sub_type
    type 固定为 "external"
    """
    prompt = f"""
You are an AI model that classifies driving-related images.

Hierarchy (for external scenes only):
{json.dumps(TYPE_HIERARCHY["external"], indent=4)}

Your task:
- Always assume type = "external".
- Identify the correct sub_type and sub_sub_type based on the visual content.
- Output strictly JSON in this format:
{{
  "type": "external",
  "sub_type": "...",
  "sub_sub_type": ["..."]
}}

If uncertain, return:
{{"type": "external", "sub_type": null, "sub_sub_type": []}}
"""

    for attempt in range(max_retries):
        try:
            image_b64 = encode_image(image_path)
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ],
                }],
                stream=False,
            )

            content = response.choices[0].message.content.strip()
            result = json.loads(content)

            if not isinstance(result, dict) or "sub_type" not in result:
                continue
            if not isinstance(result.get("sub_sub_type"), list):
                result["sub_sub_type"] = [result["sub_sub_type"]]
            return result

        except Exception as e:
            print(f"[Error] classify_image failed for {image_path} (attempt {attempt+1}): {e}")
            time.sleep(2)

    return None

# =========================================================
# 异步执行封装
# =========================================================
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def classify_image_async(image_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, classify_image, image_path)

# =========================================================
# 数据加载与按比例抽样
# =========================================================
def load_and_sample_datasets():
    json_files = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.endswith(".json")
    ]
    stats = []
    datasets = {}
    total_images = 0

  
    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data)
        datasets[jf] = data
        stats.append((jf, count))
        total_images += count

    filtered = {jf: c for jf, c in stats if c > 10_000}
    total_high = sum(filtered.values())

    sampled_images = []
    sampling_stats = []

    for jf, count in filtered.items():
        ratio = count / total_high
        sample_num = int(TARGET_TOTAL * ratio)
        data = datasets[jf]
        sampled = random.sample(data, min(sample_num, len(data)))
        sampled_images.extend(sampled)
        sampling_stats.append((jf, count, len(sampled), ratio))

    
    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Total JSON files: {len(stats)}\n")
        f.write(f"Total images (all files): {total_images}\n")
        f.write(f"Total large datasets (>10k): {len(filtered)}\n")
        f.write(f"Total images considered for sampling: {total_high}\n")
        f.write(f"Target sample total: {TARGET_TOTAL}\n\n")

        f.write("===== Dataset Counts =====\n")
        for jf, count in stats:
            f.write(f"{os.path.basename(jf)}: {count} images\n")

        f.write("\n===== Sampling Summary =====\n")
        for jf, total, sampled, ratio in sampling_stats:
            f.write(f"{os.path.basename(jf)} | total={total}, sampled={sampled}, ratio={ratio:.4f}\n")

        overall_ratio = len(sampled_images) / total_high
        f.write(f"\nOverall sampling ratio: {overall_ratio:.4%}\n")

    print(f"[Info] Sampled {len(sampled_images)} images from {len(filtered)} datasets.")
    return sampled_images

# =========================================================
# 主流程（支持断点续跑）
# =========================================================
async def main():
    sampled_images = load_and_sample_datasets()

    # ====== 加载已完成结果（断点续跑） ======
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            try:
                output_results = json.load(f)
            except:
                output_results = []
    else:
        output_results = []

    processed = {entry["image_path"] for entry in output_results}
    tasks_to_process = [p for p in sampled_images if p not in processed]
    print(f"[Resume] Already processed {len(processed)} images, will skip them.")
    print(f"[Start] Need to classify {len(tasks_to_process)} images...\n")

    sem = asyncio.Semaphore(MAX_WORKERS)
    buffer = []

    async def process_image(img_path):
        async with sem:
            result = await classify_image_async(img_path)
            if result:
                entries = []
                for sub_sub in result["sub_sub_type"]:
                    entry = {
                        "image_path": img_path,
                        "type": "external",
                        "sub_type": result["sub_type"],
                        "sub_sub_type": sub_sub
                    }
                    entries.append(entry)
                    print(f"{img_path} -> external/{result['sub_type']}/{sub_sub}")
                return entries
            return []

    tasks = [asyncio.create_task(process_image(p)) for p in tasks_to_process]
    progress_bar = async_tqdm(total=len(tasks), desc="Classifying images", ncols=100)

    for future in asyncio.as_completed(tasks):
        entries = await future
        if entries:
            buffer.extend(entries)

        if len(buffer) >= BATCH_SIZE:
            output_results.extend(buffer)
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(output_results, f, indent=4, ensure_ascii=False)
            buffer.clear()

        progress_bar.update(1)

    progress_bar.close()

    # 写入剩余部分
    if buffer:
        output_results.extend(buffer)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=4, ensure_ascii=False)

    print(f"\n Classification completed. {len(output_results)} entries saved to {OUTPUT_PATH}.")
    print(f" Dataset stats written to {STATS_PATH}")

# =========================================================
# 程序入口
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())
