from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("hf_tokenizer")
print(tok.tokenize("这是一个测试。Hello world!"))
