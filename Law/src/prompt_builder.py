def create_template(item , tokenizer):
    format_context = f"{tokenizer.bos_token}system\n{item["instruction"]}{tokenizer.eos_token}\n{tokenizer.bos_token}user\n{item["input"]}{tokenizer.eos_token}\n{tokenizer.bos_token}assistant\n{item['output']}{tokenizer.eos_token}\n"
    return format_context