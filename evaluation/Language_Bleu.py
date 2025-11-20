import json
import os
import argparse
# 确保您已安装 pycocoevalcap
# pip install pycocoevalcap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

# Scorer 类保持不变
class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref  # 预测答案 (res)
        self.gt = gt    # 标准答案 (gts)
        print('Setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
        ]
    
    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('Computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = {f"Bleu_{i+1}": s for i, s in enumerate(score)}
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score
        
        print('\n*****DONE*****')
        print(json.dumps(total_scores, indent=4))
        return total_scores

# --- 主逻辑 ---

def main():
    # 1. 设置命令行参数解析器，用于接收文件路径
    parser = argparse.ArgumentParser(description="Evaluate model predictions from a JSON or JSONL file.")
    parser.add_argument('-f', '--file',default="./model_inference/final_inferenced/DriveMMO_merged.json" ,type=str, help='Path to the input data file (must be .json or .jsonl format).')
    args = parser.parse_args()

    # 2. 从指定的文件路径加载数据
    input_file_path = args.file
    print(f"Loading data from: {input_file_path}")

    # 根据文件扩展名选择不同的加载方式
    if input_file_path.endswith('.json'):
        with open(input_file_path, 'r', encoding='utf-8') as f:
            outputs = json.load(f)
    elif input_file_path.endswith('.jsonl'):
        outputs = []
        with open(input_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                outputs.append(json.loads(line.strip()))
    else:
        raise ValueError("Unsupported file format. Please provide a .json or .jsonl file.")

    print(f"Successfully loaded {len(outputs)} items from the file.")

    # 3. 整理数据为 pycocoevalcap 要求的格式
    all_gts = {i: [item['ground_truth']] for i, item in enumerate(outputs)}
    all_preds = {i: [item['predicted_answer']] for i, item in enumerate(outputs)}

    # 4. 初始化 Scorer 并计算指标
    scorer = Scorer(all_preds, all_gts)
    metrics = scorer.compute_scores()

    # 5. 基于输入文件名，自动生成结果文件名
    base_name = os.path.basename(input_file_path)
    stem, _ = os.path.splitext(base_name)
    # 结果文件将保存在与输入文件相同的目录下
    results_file_name = os.path.join(os.path.dirname(input_file_path), f"{stem}_metrics.json")
    print(metrics)
    # print(f"\nSaving metrics to {results_file_name}...")
    # with open(results_file_name, 'w', encoding='utf-8') as f:
    #     json.dump(metrics, f, indent=4, sort_keys=True)

    # print("Metrics saved successfully.")

if __name__ == '__main__':
    main()