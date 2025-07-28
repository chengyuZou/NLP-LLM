import argparse
import json
from pathlib import Path
import joblib
import sentencepiece as spm
from sklearn.metrics import accuracy_score


def load_data(jsonl_path):
    texts, labels = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            texts.append(sample['text'])
            labels.append(sample['label'])
    return texts, labels


def main(args):
    # 1. 加载分类器 Pipeline
    ckpt_dir = Path(args.ckpt)
    model_path = ckpt_dir / 'classifier_pipeline.joblib'
    clf = joblib.load(model_path)

    # 2. 加载 SentencePiece 分词模型
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # 3. 读取测试数据
    test_jsonl = args.test_jsonl
    texts, labels = load_data(test_jsonl)

    # 4. 对文本进行分词（空格分隔字符串）
    tokenized = [" ".join(sp.encode(t, out_type=str)) for t in texts]

    # 5. 预测并计算准确率
    preds = clf.predict(tokenized)
    acc = accuracy_score(labels, preds)

    # 6. 输出结果
    print(f"Accuracy: {acc:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate trained classifier on test set")
    parser.add_argument(
        '--ckpt', type=str, required=True,
        help='Path to classifier checkpoint directory'
    )
    parser.add_argument(
        '--tokenizer', type=str, required=True,
        help='Path to SentencePiece .model file used during training'
    )
    parser.add_argument(
        '--test_jsonl', type=str, default='holdout.jsonl',
        help='Test data JSONL file path (with text and label)'
    )

    args = parser.parse_args()
    main(args)
