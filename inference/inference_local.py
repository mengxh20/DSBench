#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import json
import time
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# ----------------------------
# Configuration
# ----------------------------
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "8"))

client = OpenAI(
    api_key="", 
    base_url="http://10.82.16.125:8000/v1/",  
)

# ----------------------------
# Image encoding helper
# ----------------------------
def encode_image(image_path):
    """读取图片并转base64"""
    pil = Image.open(image_path).convert("RGB")
    buffered = BytesIO()
    pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# ----------------------------
# Gateway inference
# ----------------------------
def run_inference(model_name, question, image_path):
    try:
        img_base64 = encode_image(image_path)

        messages = [
            {"role": "system", "content": "You are a helpful assistant for autonomous driving safety reasoning."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                ],
            },
        ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            stream=False,  
        )

        result = response.choices[0].message.content.strip()
        return result

    except Exception as e:
        print(f" Error during inference on {image_path}: {e}")
        return ""

# ----------------------------
# Process dataset
# ----------------------------
def main(model_name, input_path, output_dir, num_workers=NUM_WORKERS):
    # 读取输入文件
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f" Loaded {len(data)} samples from {input_path}")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.json")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        futures = {
            exe.submit(run_inference, model_name, sample["question"], sample["image_path"]): sample
            for sample in data
        }

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Model Inference"):
            sample = futures[fut]
            predicted = fut.result()
            sample["predicted_answer"] = predicted
            results.append(sample)

    # 保存结果
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f" Inference finished. Results saved to:\n{output_path}")

# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf" 
    input_path = "/dataset_path/DSBench_QAs.json"
    output_dir = "./model_inference"

    main(model_name, input_path, output_dir)
