import json
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

def load_texts_from_jsonl(jsonl_path):
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            text = sample.get("text")
            if text:
                yield text

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("Train HF-compatible BPE tokenizer from JSONL corpus")
    parser.add_argument("--jsonl", type=str, default = "THUCNews.jsonl", help="输入 JSONL 文件，含 'text'")
    parser.add_argument("--vocab_size", type=int, default=16000, help="词表大小")
    parser.add_argument("--output_dir", type=str, default="hf_tokenizer", help="输出目录")
    args = parser.parse_args()

    # 1. 初始化 BPE tokenizer，指定 unk_token
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # 2. 指定所有需要的 special tokens
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=special_tokens
    )

    # 3. 使用 JSONL 文本训练 tokenizer
    tokenizer.train_from_iterator(load_texts_from_jsonl(args.jsonl), trainer=trainer)

    # 4. 确保 special tokens 已经在 vocab 里
    cls_id = tokenizer.token_to_id("[CLS]")
    sep_id = tokenizer.token_to_id("[SEP]")
    pad_id = tokenizer.token_to_id("[PAD]")
    unk_id = tokenizer.token_to_id("[UNK]")
    mask_id = tokenizer.token_to_id("[MASK]")
    for tok, tid in [("[CLS]", cls_id), ("[SEP]", sep_id), ("[PAD]", pad_id),
                     ("[UNK]", unk_id), ("[MASK]", mask_id)]:
        if tid is None:
            raise ValueError(f"Special token {tok} missing from vocab!")

    # 5. 配置 post-processor，使用 tuple (token, id) 格式
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", cls_id),
            ("[SEP]", sep_id),
        ]
    )

    # 6. 保存 tokenizer.json 和 HuggingFace 格式
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tk_path = output_dir / "tokenizer.json"
    tokenizer.save(str(tk_path))

    hf = PreTrainedTokenizerFast(
        tokenizer_file=str(tk_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    hf.save_pretrained(str(output_dir))

    print(f"HF tokenizer saved to {output_dir}")
