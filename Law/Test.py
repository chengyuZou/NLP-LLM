from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import torch
OUTPUT_DIR = "./lora_legal_qa_adapter"
SAVE_STEPS = 500           # 保存步数（示例）
EVAL_STEPS = 500           # 评估步数，必须与 save 策略/间隔一致以使用 load_best_model_at_end

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/", use_fast=False, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 数据处理函数
def process_function(examples):
    input_ids, attention_mask, labels = [], [], []
    max_length = 1024
    
    for i in range(len(examples["input"])):
        # 从instruction与input中生成input_ids
        inputs = "\n Human: " + examples["instruction"][i].strip() + examples["input"][i].strip() + "\n\nAssistant:"
        instruction = tokenizer(inputs, add_special_tokens=False)
        re = examples["output"][i] + tokenizer.eos_token
        if examples["output"][i] is not None:
            response = tokenizer(re, add_special_tokens=False)
        else:
            continue
        
        input_id = instruction["input_ids"] + response["input_ids"]
        att_mask = instruction["attention_mask"] + response["attention_mask"]
        label = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        
        # 截断到最大长度
        if len(input_id) > max_length:
            input_id = input_id[:max_length]
            att_mask = att_mask[:max_length]
            label = label[:max_length]
        elif len(input_id) < max_length:
            # 填充到相同长度
            padding_length = max_length - len(input_id)
            input_id.extend([tokenizer.pad_token_id] * padding_length)
            att_mask.extend([0] * padding_length)
            label.extend([-100] * padding_length)
            
        input_ids.append(input_id)
        attention_mask.append(att_mask)
        labels.append(label)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

# 加载数据
dataset = load_dataset("./SFT/LoRA")
dataset = dataset['train'].map(process_function, batched=True, remove_columns=dataset['train'].column_names, 
                              batch_size=4)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# 创建数据收集器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "./Baichuan2-7B-Base/", 
    torch_dtype=torch.float16, 
    device_map="auto", 
    trust_remote_code=True
)

# 设置 LoRA 配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False,
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1, 
    target_modules=["o_proj", "gate_proj", "down_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# TrainingArguments：evaluation_strategy 必须和 save_strategy/steps 配合
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=10,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=100,
    save_steps=SAVE_STEPS,
    save_strategy="steps",
    eval_strategy="steps",   # <-- 必须启用 evaluation
    eval_steps=EVAL_STEPS,        # <-- 与 save_steps/strategy 保持一致或合理设置
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",  # 使用 eval_loss 作为选择指标
    greater_is_better=False,
    warmup_steps=100,
    max_grad_norm=1.0,
    report_to=None,
    ddp_find_unused_parameters=False,
)

# 创建 Trainer（注意传入 eval_dataset）
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 开始训练
trainer.train()

# 保存模型
trainer.save_model()

# 保存最终模型
trainer.save_state()