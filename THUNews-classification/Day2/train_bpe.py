import sentencepiece as spm
import os

input_file = "/root/autodl-tmp/all_texts.txt"
vocab_size = 32000
model_type = "bpe"
model_prefix = f"spm_bpe_{vocab_size}"
output_dir = "/root/autodl-tmp/thucnews_subword/"

# 构建训练命令
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=os.path.join(output_dir, model_prefix),
    vocab_size=vocab_size,
    model_type=model_type,
    character_coverage=0.9995,
    unk_id=0,
    pad_id=1,
    bos_id=2,
    eos_id=3
)

print(f"✅ 训练完成: {model_prefix}.model 已保存到 {output_dir}")
