import sentencepiece as spm

# 1. 实例化并加载模型
sp = spm.SentencePieceProcessor()
sp.load("spm_bpe_8000.model")  # 只需 .model 文件

# 2. （可选）查看特殊符号 ID
print("unk_id =", sp.unk_id())
print("pad_id =", sp.pad_id())
print("bos_id =", sp.bos_id())
print("eos_id =", sp.eos_id())

# 3. 分词测试
pieces = sp.encode("这是中文测试，Hello world!", out_type=str)
ids    = sp.encode("这是中文测试，Hello world!", out_type=int)
print(pieces)
print(ids)
