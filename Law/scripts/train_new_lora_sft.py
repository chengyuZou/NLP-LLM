# train_lora_from_new_sft.py
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import torch

from src.config import Config


# 1) 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(Config.baichuan_model_path, use_fast=False, trust_remote_code=True)
# 确保有 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 加载原始 jsonl（每行 dict，包含 id, reference(list), input, output）
raw = load_dataset("json", data_files={"train": Config.triplet_data_path}, split="train")
print("raw sample count:", len(raw))

# 3) 将原始样本转为 SFT prompt/response 格式
#    我这里采用：prompt 包含已知法律条文(reference) + 问题(input)
#    response 就是 output（你的 ground-truth 回答）
def make_prompt(example):

    # 构造 prompt（你之前训练时使用了 "Human/Assistant" 模式）
    prompt_parts = []
    prompt_parts.append("你是一名专业律师,请根据中国法律并基于以上条文,回答下列问题：")
    prompt_parts.append(f"问题：{example['input'].strip()}")
    # 最终 prompt
    prompt = "\n".join(prompt_parts)
    # 目标输出
    response = example.get("output", "").strip()
    # 确保 response 以 eos_token 结束（有利于 label 处理）
    if not response.endswith(tokenizer.eos_token or ""):
        response = response + (tokenizer.eos_token or "")
    return {"prompt": prompt, "response": response, "id": example.get("id","")}

# map 生成 prompt/response（非 tokenized）
pairwise = raw.map(lambda ex: make_prompt(ex), batched=False)
print("示例转换完成,前2条:")
for i in range(min(2, len(pairwise))):
    print(i, pairwise[i])

# 4) tokenization + 生成 labels（prompt 部分为 -100）
#    使用 batched=True，但逐条构造以便对 prompt 长度进行截断保留 response 更好（保留后缀）
def tokenize_and_build_labels(examples):
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    for p, r in zip(examples["prompt"], examples["response"]):
        # 我们采用 "Human/Assistant" 的显式包裹风格（确保 prompt tokens 不计 loss）
        full_input = "\nHuman: " + p.strip() + "\n\nAssistant: "
        # tokenize prompt & response separately (avoid double special tokens)
        enc_prompt = tokenizer(full_input, add_special_tokens=False)
        enc_resp = tokenizer(r, add_special_tokens=False)

        input_ids = enc_prompt["input_ids"] + enc_resp["input_ids"]
        attention_mask = enc_prompt["attention_mask"] + enc_resp["attention_mask"]
        labels = [-100] * len(enc_prompt["input_ids"]) + enc_resp["input_ids"]

        # 截断或填充到最大长度 MAX_LENGTH（这里保留 response 的后部优先）
        if len(input_ids) > Config.tokenizer_max_length:
            # 保留后 MAX_LENGTH tokens（以尽可能保留response）
            input_ids = input_ids[-Config.tokenizer_max_length:]
            attention_mask = attention_mask[-Config.tokenizer_max_length:]
            labels = labels[-Config.tokenizer_max_length:]
        else:
            # 不在这里 pad（交给 data_collator 做 dynamic padding），但为了避免 extremely short example 我们可以不 pad
            padding_length = Config.tokenizer_max_length - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([-100] * padding_length)
            

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

tokenized = pairwise.map(tokenize_and_build_labels, batched=True, remove_columns=pairwise.column_names , batch_size = 4)
print("tokenized example count:", len(tokenized))
print("示例 tokenized(第0条)长度:", len(tokenized[0]["input_ids"]))

# 5) train/test split
split = tokenized.train_test_split(test_size=0.1, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]
print("train/eval sizes:", len(train_ds), len(eval_ds))

# 6) Data collator（动态 pad）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 7) 加载模型 & 封装 LoRA
# 注意：如果你想使用 8-bit 加载 (bitsandbytes) 或更复杂设置，这里可替换
base_model = AutoModelForCausalLM.from_pretrained(
    Config.baichuan_model_path,
    trust_remote_code=True,
    dtype = Config.dtype,
    device_map = Config.device
)
model = PeftModel.from_pretrained(base_model, Config.apapter_output_dir)

# 确保 LoRA 层启用梯度计算
for param in model.parameters():
    param.requires_grad = True

# LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",        # 任务类型
    inference_mode=False,         # 训练模式
    r=Config.lora_r,                          # LoRA秩
    lora_alpha=Config.lora_alpha,                # LoRA缩放系数
    lora_dropout=Config.lora_dropout,             # Dropout
    target_modules=Config.lora_target_modules  # LoRA适配层
)

# 再次加载 LoRA 配置
model = get_peft_model(base_model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()

# 8) TrainingArguments + Trainer
training_args = TrainingArguments(
    output_dir = Config.new_apapter_output_dir, # 输出目录
    overwrite_output_dir = Config.overwrite_output_dir, # 覆盖输出目录中的内容
    num_train_epochs = Config.train_epochs, # 训练轮数
    per_device_train_batch_size = Config.per_device_train_batch_size, # 每个设备的训练批次大小
    per_device_eval_batch_size = Config.per_device_eval_batch_size, # 每个设备的评估批次大小
    learning_rate = Config.learning_rate, # 学习率
    gradient_accumulation_steps = Config.gradient_accumulation_steps, # 梯度累积步数（模拟更大的批次大小）
    fp16 = True, # 使用混合精度训练
    logging_steps = Config.logging_steps, # 每多少步记录一次日志
    save_steps = Config.save_steps, # 保存间隔步数
    save_strategy = Config.save_strategy, # 按步数保存策略
    eval_steps = Config.eval_steps, # 评估间隔步数
    eval_strategy = Config.eval_strategy, # 按步数评估策略
    load_best_model_at_end = True, # 训练结束时加载最佳模型
    metric_for_best_model = Config.metric_for_best_model, # 用于选择最佳模型的指标
    greater_is_better = False, # eval_loss越小越好
    warmup_steps = Config.warmup_steps, # 学习率预热步数
    max_grad_norm = Config.max_grad_norm, # 梯度裁剪的最大范数
    report_to = None, # 不向任何平台报告（如wandb等）
    ddp_find_unused_parameters = False # 在分布式训练中不查找未使用参数
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 9) 启动训练
trainer.train()

# 10) 保存 LoRA adapter（和 tokenizer info）
os.makedirs(Config.new_apapter_output_dir, exist_ok=True)
model.save_pretrained(Config.new_apapter_output_dir)
tokenizer.save_pretrained(Config.new_apapter_output_dir)
print("LoRA adapter saved to:", Config.new_apapter_output_dir)