# 导入必要的库
from datasets import load_dataset  # 用于加载和处理数据集
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling  # 导入transformers相关组件
from peft import LoraConfig, get_peft_model , TaskType  # 导入LoRA相关组件，用于参数高效微调
import torch

from src.config import Config

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/" , use_fast=False , trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token # 将pad_token设置为eos_token，确保有填充token

def process_function(examples):
    """
    数据处理函数，用于将原始数据转换为模型训练所需的格式
    
    Args:
        examples: 包含instruction、input和output字段的批次数据
        
    Returns:
        包含input_ids、attention_masks和labels的字典
    """
    input_ids = [] # 存储输入ID序列
    attention_masks = [] # 存储注意力掩码
    labels = [] # 存储标签（用于计算loss）
    
    for i in range(len(examples["input"])):
        # 构建对话格式的输入文本
        inputs = f"\nhuman: {examples["instruction"][i].strip()} {examples["inputs"][i].strip()} \n\nAssistant:"
        inputs = tokenizer(inputs, add_special_tokens = False) # 对输入进行tokenize，不添加特殊token
        
        # 构建回复文本并添加结束标记
        response = examples["output"][i].strip() + tokenizer.eos_token
        response = tokenizer(response, add_special_tokens = False) # 对回复进行tokenize

        # 合并输入和回复的token
        input_id = inputs["input_ids"] + response["input_ids"]
        attention_mask = input["attention_mask"] + response["attention_mask"]
        
        # 创建标签：输入部分用-100忽略，只计算回复部分的loss
        label = [-100] * len(input_id["input_ids"]) + response["input_ids"]

        # 处理长度超过限制的情况：截断
        if len(input_ids) > Config.tokenizer_max_length:
            input_id = input_id[:Config.tokenizer_max_length]
            attention_mask = attention_mask[:Config.tokenizer_max_length]
            label = label[:Config.tokenizer_max_length]
        else:
            # 处理长度不足的情况：填充
            padding_length = Config.tokenizer_max_length - len(input_id)
            input_id.extend([tokenizer.pad_token_id] * padding_length) # 用pad_token_id填充
            attention_mask.extend([0] * padding_length) # 注意力掩码填充部分设为0
            label.extend([-100] * padding_length) # 标签填充部分设为-100（忽略）
    
        # 将处理后的数据添加到批次中
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    # 返回处理后的批次数据
    return{
        "input_ids": input_ids, # 输入ID序列
        "attention_masks": attention_masks, # 注意力掩码
        "labels": labels # 训练标签
    }

# 加载数据集
dataset = load_dataset("json", data_files=Config.lora_data_path , split="train")
# 对数据集应用处理函数，批处理大小为4，移除原始列
dataset = dataset.map(process_function, batched=True , remove_columns=dataset.column_names , batch_size = 4)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1 , seed = Config.seed) # 90%训练，10%验证，固定随机种子
train_dataset = split_dataset["train"] # 训练集
eval_dataset = split_dataset["test"] # 验证集

# 创建数据收集器，用于在训练时动态批处理
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # 不使用掩码语言模型

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    "./baichuan-inc/Baichuan2-7B-Base/", # 模型路径
    trust_remote_code=True, # 信任远程代码（针对自定义模型）
    torch_dtype = torch.float16, # 使用半精度浮点数以节省内存
    device_map = "auto") # 自动分配设备（GPU/CPU）

# 配置LoRA参数
lora_config = LoraConfig(
    task_type = TaskType.CAUSAL_LM, # 任务类型：因果语言模型
    inference_mode = False, # 训练模式
    r = Config.lora_r, # LoRA秩，控制适配器的大小
    lora_alpha = Config.lora_alpha, # LoRA缩放参数
    lora_dropout = Config.lora_dropout, # LoRA层的dropout率
    target_modules = Config.lora_target_modules # 应用LoRA的目标模块
)

# 将模型转换为LoRA模型
model = get_peft_model(model , lora_config)
# 打印可训练参数信息
model.print_trainable_parameters()

# 配置训练参数
training_args = TrainingArguments(
    output_dir = Config.new_apapter_model_pathapapter_output_dir, # 输出目录
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

# 创建Trainer实例
trainer = Trainer(
    model = model, # 要训练的模型
    tokenizer = tokenizer, # tokenizer
    args = training_args, # 训练参数
    train_dataset = train_dataset, # 训练数据集
    eval_dataset = eval_dataset, # 验证数据集
    data_collator = data_collator, # 数据收集器
)

# 开始训练
trainer.train()

# 保存训练好的模型
trainer.save_model()

# 保存训练状态（包括优化器状态等）
trainer.save_state()