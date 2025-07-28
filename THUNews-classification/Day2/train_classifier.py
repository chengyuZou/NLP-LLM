import argparse
import json
from pathlib import Path
import sentencepiece as spm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib


def load_data(jsonl_path):
    texts, labels = [], []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            texts.append(sample['text'])
            labels.append(sample.get('label'))
    return texts, labels


def main(args):
    # 1. 加载 SentencePiece 模型
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer)

    # 2. 读取训练数据
    print(args.train_jsonl)
    texts, labels = load_data(args.train_jsonl)

    # 3. 分词并准备特征：先将 tokens 列表转为空格分隔的字符串
    tokenized_texts = [" ".join(sp.encode(t, out_type=str)) for t in texts]

    # 4. 构建分类器 Pipeline（CountVectorizer + LogisticRegression）
    clf = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', LogisticRegression(max_iter=5))
    ])

    # 5. 训练模型
    print("Training classifier...")
    clf.fit(tokenized_texts, labels)

    # 6. 保存模型
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'classifier_pipeline.joblib'
    joblib.dump(clf, model_path)
    print(f"Model saved to {model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a text classifier using SentencePiece tokenization")
    parser.add_argument(
        '--tokenizer', type=str, required=True,
        help='Path to the SentencePiece .model file'
    )
    parser.add_argument(
        '--train_jsonl', type=str, required=True,
        help="Training data in JSONL format, each line with {'text':..., 'label':...}"
    )
    parser.add_argument(
        '--output_dir', type=str, required=True,
        help='Directory to save the trained classifier'
    )
    args = parser.parse_args()
    main(args)
