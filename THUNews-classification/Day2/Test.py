import os
import subprocess
import argparse
import json
from pathlib import Path
import sentencepiece as spm

def load_tokenizer(model_file: str):
    sp = spm.SentencePieceProcessor()
    sp.load(model_file)
    return sp

def compute_avg_length_and_oov(sp: spm.SentencePieceProcessor, path_jsonl: str):
    total_tokens = 0
    total_samples = 0
    oov_count = 0

    with open(path_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample['text']
            ids = sp.encode(text, out_type=int)
            total_tokens += len(ids)
            total_samples += 1
            oov_count += sum(1 for i in ids if i == sp.unk_id())

    avg_len = total_tokens / total_samples
    oov_rate = oov_count / total_tokens
    return avg_len, oov_rate

def main(args):
    sizes       = args.vocab_sizes
    models_dir  = Path(args.models_dir)
    train_jsonl = args.train_jsonl
    holdout_jsonl = args.holdout_jsonl
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    print(f"sizes： {sizes} \
           models_dir： {models_dir} \
           train_jsonl： {train_jsonl} \
           holdout_jsonl：{holdout_jsonl} \
           output_root: {output_root} "
         )
           

    results = []

    for size in sizes:
        print(f"\n===== vocab_size = {size} =====")

        # 1. 直接加载已有的 SPM 模型
        model_file = models_dir / f"spm_bpe_{size}.model"
        if not model_file.exists():
            raise FileNotFoundError(f"Cannot find model file: {model_file}")
        sp = load_tokenizer(str(model_file))
        print(f"Loaded SPM model -> {model_file}")

        # 2. 计算 avg length & OOV rate
        avg_len, oov_rate = compute_avg_length_and_oov(sp, holdout_jsonl)
        print(f"Avg tokens/sample = {avg_len:.2f}, OOV rate = {oov_rate:.4f}")

        # 3. 训练分类器
        ckpt_dir = output_root / f"ckpt_bpe_{size}"
        ckpt_dir.mkdir(exist_ok=True)
        subprocess.run([
            "python", "train_classifier.py",
            "--tokenizer", str(model_file),
            "--train_jsonl", train_jsonl,
            "--output_dir", str(ckpt_dir)
        ], check=True)

        # 4. 评估分类器
        eval_res = subprocess.run([
            "python", "eval_classifier.py",
            "--ckpt", str(ckpt_dir),
            "--tokenizer", str(model_file),
            "--test_jsonl", holdout_jsonl
        ], capture_output=True, text=True, check=True)
        # 假设 eval 脚本最后输出一行: "Accuracy: 0.XXXX"
        acc = float(eval_res.stdout.strip().split()[-1])
        print(f"Classification accuracy = {acc:.4f}")

        results.append({
            "vocab_size": size,
            "avg_len": avg_len,
            "oov_rate": oov_rate,
            "accuracy": acc
        })

    # 保存所有实验结果
    with open(output_root / "vocab_size_experiment_results.json", 'w', encoding='utf-8') as fw:
        json.dump(results, fw, ensure_ascii=False, indent=2)

    print("\nAll experiments done. Results saved to vocab_size_experiment_results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vocab-size ablation for BPE")
    parser.add_argument(
        "--vocab_sizes", nargs='+', type=int, default=[8000,16000,32000],
        help="要对比的 vocab sizes 列表"
    )
    parser.add_argument(
        "--models_dir", type=str, default="",
        help="存放 spm_bpe_{size}.model/.vocab 的目录"
    )
    parser.add_argument(
        "--train_jsonl", type=str, default="train.jsonl",
        help="训练集 JSONL 文件路径（包含 'text' 字段）"
    )
    parser.add_argument(
        "--holdout_jsonl", type=str, default="holdout.jsonl",
        help="留出集 JSONL 文件路径，用于计算 avg len 和 OOV"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiments",
        help="实验结果和 checkpoint 输出根目录"
    )
    args = parser.parse_args()

    main(args)
