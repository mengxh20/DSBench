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
DATASET_DIR = "/e2e-data/evad-tech-vla/mxh24/AD-SAFETY/datasets/cabin"
OUTPUT_PATH = "classified_images_cabin.json"
STATS_PATH = "dataset_stats_cabin.txt"
MAX_WORKERS = 10
BATCH_SIZE = 50

# =========================================================
# 分类层级定义（cabin）
# =========================================================
TYPE_HIERARCHY = {
    "cabin": {
        "Emotion": ["Peace", "Anxiety", "Anger", "Weariness", "Happiness"],
        "Attention": ["Distracted", "Exhaustion", "Looking Around"],
        "Driver operation": [
            "Hands off wheel",
            "Using phone",
            "Calling",
            "Smoking"
        ],
        "Cockpit environment": [
            "No seat belt",
            "Seat belt on",
            "Mirror dirty"
        ]
    }
}

# =========================================================
# 图像分类函数（支持多个 sub_type）
# =========================================================
def encode_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_image(image_path, max_retries=3):
    """
    分类函数：
    如果模型返回空结果(results: [])，则默认：
    {
      "type": "cabin",
      "results": [
         {"sub_type": "Emotion", "sub_sub_type": ["Calm down"]}
      ]
    }
    """
    prompt = f"""
You are an AI model that classifies in-cabin (driver monitoring) images.

Hierarchy (for cabin scenes only):
{json.dumps(TYPE_HIERARCHY["cabin"], indent=4)}

Your task:
- Always assume type = "cabin".
- Identify **all possible** sub_type and sub_sub_type combinations present in the image.
- Output strictly JSON in this format:
{{
  "type": "cabin",
  "results": [
     {{"sub_type": "...", "sub_sub_type": ["..."]}},
     {{"sub_type": "...", "sub_sub_type": ["...", "..."]}}
  ]
}}

If uncertain, return:
{{"type": "cabin", "results": []}}
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

            # 校验格式
            if not isinstance(result, dict) or "results" not in result:
                continue

            valid_results = []
            for r in result["results"]:
                if not isinstance(r, dict) or "sub_type" not in r:
                    continue
                sub_sub_list = r.get("sub_sub_type", [])
                if isinstance(sub_sub_list, str):
                    sub_sub_list = [sub_sub_list]
                valid_results.append({
                    "sub_type": r["sub_type"],
                    "sub_sub_type": sub_sub_list
                })

            # ✅ 修改点：如果结果为空，填默认值 Calm down
            if not valid_results:
                valid_results = [{
                    "sub_type": "Emotion",
                    "sub_sub_type": ["Calm down"]
                }]

            result["results"] = valid_results
            return result

        except Exception as e:
            print(f"[Error] classify_image failed for {image_path} (attempt {attempt+1}): {e}")
            time.sleep(2)

    # 如果多次重试仍失败，返回默认 Calm down
    return {
        "type": "cabin",
        "results": [
            {"sub_type": "Emotion", "sub_sub_type": ["Calm down"]}
        ]
    }


# =========================================================
# 异步执行封装
# =========================================================
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

async def classify_image_async(image_path):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, classify_image, image_path)


# =========================================================
# 数据加载
# =========================================================
def load_all_datasets():
    json_files = [
        os.path.join(DATASET_DIR, f)
        for f in os.listdir(DATASET_DIR)
        if f.endswith(".json")
    ]

    stats = []
    all_images = []

    for jf in json_files:
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
        count = len(data)
        all_images.extend(data)
        stats.append((jf, count))

    total_images = len(all_images)

    with open(STATS_PATH, "w", encoding="utf-8") as f:
        f.write(f"Total JSON files: {len(stats)}\n")
        f.write(f"Total images (all files): {total_images}\n\n")
        f.write("===== Dataset Counts =====\n")
        for jf, count in stats:
            f.write(f"{os.path.basename(jf)}: {count} images\n")

    print(f"[Info] Loaded {total_images} images from {len(stats)} datasets (no sampling).")
    return all_images


# =========================================================
# 主流程
# =========================================================
async def main():
    all_images = load_all_datasets()

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            try:
                output_results = json.load(f)
            except:
                output_results = []
    else:
        output_results = []

    processed = {entry["image_path"] for entry in output_results}
    tasks_to_process = [p for p in all_images if p not in processed]
    print(f"[Resume] Already processed {len(processed)} images, will skip them.")
    print(f"[Start] Need to classify {len(tasks_to_process)} images...\n")

    sem = asyncio.Semaphore(MAX_WORKERS)
    buffer = []

    async def process_image(img_path):
        async with sem:
            result = await classify_image_async(img_path)
            print(f"[Raw result] {result}")
            entries = []
            for r in result.get("results", []):
                sub_type = r.get("sub_type")
                for sub_sub in r.get("sub_sub_type", []):
                    entry = {
                        "image_path": img_path,
                        "type": "cabin",
                        "sub_type": sub_type,
                        "sub_sub_type": sub_sub
                    }
                    entries.append(entry)
                    print(f"{img_path} -> cabin/{sub_type}/{sub_sub}")
            return entries

    tasks = [asyncio.create_task(process_image(p)) for p in tasks_to_process]
    progress_bar = async_tqdm(total=len(tasks), desc="Classifying cabin images", ncols=100)

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

    if buffer:
        output_results.extend(buffer)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=4, ensure_ascii=False)

    print(f"\n[Done] Classification completed. {len(output_results)} entries saved to {OUTPUT_PATH}.")
    print(f"[Info] Dataset stats written to {STATS_PATH}")


# =========================================================
# 程序入口
# =========================================================
if __name__ == "__main__":
    asyncio.run(main())
