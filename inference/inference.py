#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from tqdm import tqdm
from PIL import Image
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# =====================================================
# 基础配置
# =====================================================
input_path = "/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/ADqa_output_benchamrk_rewritten_textonly_belt.json"
output_dir = "/e2e-data/evad-tech-vla/zhangyuchen/ADqa_generation/model_inference"
os.makedirs(output_dir, exist_ok=True)

# 需要推理的模型路径
model_paths = [
    # "/e2e-data/evad-osc-datasets/pretrained/Qwen/Qwen2.5-VL-7B-Instruct/",
    # "/e2e-data/evad-osc-datasets/pretrained/MiMo/MiMo-VL-7B-RL-2508/",
    # "/e2e-data/evad-osc-datasets/pretrained/MiMo/MiMo-VL-7B-SFT-2508/",
    # "/e2e-data/embodied-research-data/luzheng/cvpr/saves/qwen2_5vl-7b/fulltrain/checkpoint-6600",
    # "/e2e-data/embodied-research-data/luzheng/cvpr/saves/qwen2_5vl-7b/fulltrain/checkpoint-7730",  # 这个是我们最终用的DSVLM权重
    "/e2e-data/embodied-research-data/luzheng/cvpr/saves/qwen3vl-8b/fulltrain/checkpoint-200",
]

# 推理参数
gen_args = {
    "temperature": 0,
    "top_p": 1,
    "max_new_tokens": 1024,
    "repetition_penalty": 1.05,
    "do_sample": False,
}

use_flash_attention = False
max_pixels = 1605632
min_pixels = 256 * 28 * 28

# =====================================================
# 辅助函数
# =====================================================
def is_yes_no_question(question: str) -> bool:
    """判断是否是判断题（Yes/No类型问题）"""
    q_lower = question.lower().strip()
    yes_no_keywords = [
        "is ", "are ", "was ", "were ", "can ", "should ", "could ", "would ",
        "does ", "do ", "did ", "has ", "have ", "had "
    ]
    return q_lower.endswith("?") and any(q_lower.startswith(k) for k in yes_no_keywords)


def load_dataset(path):
    print(f">>> Loading input dataset: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f">>> Loaded {len(data)} samples.\n")
    return data


def load_checkpoint(output_path):
    """加载断点文件"""
    results = []
    processed_paths = set()
    if os.path.exists(output_path):
        print(f">>> Found existing output file: {output_path}")
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                results = json.load(f)
                for r in results:
                    if "image_path" in r:
                        processed_paths.add(r["image_path"])
            print(f">>> Resuming from checkpoint — already processed {len(processed_paths)} items.\n")
        except Exception as e:
            print(f"⚠️ Failed to load previous results, starting fresh: {e}")
            results = []
    else:
        print(">>> No previous results found, starting from scratch.\n")
    return results, processed_paths


def save_results(path, results):
    """实时保存结果"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


# =====================================================
# 主推理流程
# =====================================================
dataset = load_dataset(input_path)

for model_path in model_paths:
    model_name = os.path.basename(model_path.rstrip("/"))  # 取最后一级目录作为模型名
    output_path = os.path.join(output_dir, f"{model_name}.json")

    print("===================================================")
    print(f"🧠 Starting inference for model: {model_name}")
    print("===================================================")

    # ---------------------------
    # 加载模型与处理器
    # ---------------------------
    print(">>> Loading model and processor...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2" if use_flash_attention else None,
    )
    processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels, min_pixels=min_pixels)
    print(">>> Model loaded successfully.\n")

    # ---------------------------
    # 加载断点
    # ---------------------------
    results, processed_paths = load_checkpoint(output_path)

    # ---------------------------
    # 推理循环
    # ---------------------------
    for idx, sample in enumerate(tqdm(dataset, desc=f"Inference ({model_name})")):
        image_path = sample.get("image_path")
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")

        if image_path in processed_paths:
            continue

        if not image_path or not os.path.exists(image_path):
            print(f"[{idx}] ⚠️ Invalid image path: {image_path}")
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"[{idx}] ⚠️ Failed to load image: {e}")
            continue

        # 构造 prompt
        if is_yes_no_question(question):
            prompt = (
                f"This is a yes/no question. Determine whether the statement is true or false, "
                f"and explain briefly.\nQuestion: {question}"
            )
        else:
            prompt = f"Answer the following question clearly and concisely:\n{question}"

        messages = [
            {"role": "system", "content": "You are a helpful assistant for autonomous driving safety analysis."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        # 模型推理
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, **gen_args)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

        # 保存结果
        result_item = dict(sample)
        result_item["predicted_answer"] = output_text
        results.append(result_item)
        processed_paths.add(image_path)

        print(f"[{idx+1}/{len(dataset)}]  Q: {question}")
        print(f"   Pred: {output_text}\n")

        # 实时保存（每20条一次）
        if (idx + 1) % 20 == 0:
            save_results(output_path, results)

    # ---------------------------
    # 保存最终结果
    # ---------------------------
    save_results(output_path, results)
    print(f"\n✅ Inference complete for model: {model_name}")
    print(f"   Results saved to: {output_path}\n")

print("===================================================")
print("🎯 All model inferences completed successfully.")
print("===================================================")
