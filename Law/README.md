# 基于BaiChuan2-7B的法律微调大模型+RAG问答

技术栈：

• 实现了完整的pipeline，包括数据预处理、模型训练、评估和部署。

• 深度学习框架：PyTorch, Transformers, PEFT

 • 语言模型：Baichuan2-7B-Chat
 
 • 向量数据库：Faiss
 
 • 检索增强生成：LangChain, RAG
 
 • 嵌入模型：bge-large-zh

 

## 1.数据集
所选数据集为
**[法律QA](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT), 其中取 DISC-Law-SFT-Pair-QA-released.jsonl 这一项**
下载完后放在当前文件夹后解压

### 1.1 AutoDL平台下载HF数据
若无法下载或显示无法连接到huggingface.co

请参考[CSDN博客](https://blog.csdn.net/Katherine_java/article/details/146294013?ops_request_misc=&request_id=&biz_id=102&utm_term=AutoDL%E8%BF%9E%E6%8E%A5HF&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-146294013.142^v102^pc_search_result_base1&spm=1018.2226.3001.4187)


先在终端输入 source /etc/network_turbo 进行学术资源加速。 


再下载 hfd 脚本 依次输入 wget https://hf-mirror.com/hfd/hfd.sh 与 chmod a+x hfd.sh  # 赋予执行权限。


最后设置镜像环境变量，输入 export HF_ENDPOINT=https://hf-mirror.com  # 临时生效

### 1.2 AutoDL平台下载包
先更新,终端输入sudo apt update 
安装git-lfs ， 输入 sudo apt install git-lfs
然后可以用HF官网给出的脚本下载

可以用以下两段代码查看数据
```python
import json
file_path = "your_path" #你DISC-Law-SFT-Pair-QA-released.jsonl的路径
data = []
with open (file_path , 'r' , encoding='utf-8') as f:  #读取整个文件
    for line in f:  #遍历每一行
        line = json.loads(line) #以json形式每行读取
        data.append(line.strip()) #添加到List容器中， strip()函数为调整间距，去掉空格等
print(len(data))
print(data[-1])
print(data[0])
```

```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="DISC-Law-SFT-Pair-QA-released.jsonl" , split="train")
print(dataset)
```
格式
```python
Dataset({
    features: ['id', 'input', 'output'],
    num_rows: 246450
})
```

这一段代码需要你运行，在终端输入 python Data_Process.py 构建Prompt训练数据
```python
import json
from datasets import load_dataset
with open("DISC-Law-SFT-Pair-QA-released.jsonl" , 'r' , encoding='utf-8') as input_file , open("./LoRA_data.jsonl" , 'w', encoding='utf-8') as output_file:
    for line in input_file:
        data = json.loads(line.strip())
        LoRA_data = {
                "instruction": "你是一名专业律师，请根据中国法律回答以下问题。",
                "input": data["input"],
                "output": data["output"]
        }

        output_file.write(json.dumps(LoRA_data ,  ensure_ascii=False) + "\n")

data = load_dataset("json", data_files="LoRA_data.jsonl" , split="train")
data = data['train']
print(data[0])
```

```json
{'instruction': '你是一名专业律师，请根据中国法律回答以下问题。',
 'input': '违章停车与违法停车是否有区别？',
 'output': '对违反道路交通安全法律、法规关于机动车停放、临时停车规定的，可以指出违法行为，并予以口头警告，令其立即驶离。机动车驾驶人不在现场或者虽在现场但拒绝立即驶离，妨碍其他车辆、行人通行的处二十元以上二百元以下罚款。现在人们大多是称作违法停车，因此在法律责任上也会更多一些，不要以为违反交通规章制度问题不大，不要认为违法停车是罚款而已。'}

```

## 2. Base模型与Tokenizer
选用[BaiChuan2-7B](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
总大小约15GB

### 2.1 AutoDL平台下载
与 1.1和1.2同理

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#加载模型
model = AutoModelForCausalLM.from_pretrained(
    "./baichuan-inc/Baichuan2-7B-Base/",
    trust_remote_code=True,
    torch_dtype = torch.float16,
    device_map = "auto")
#加载分词器
tokenizer = AutoTokenizer.from_pretrained(
    "./baichuan-inc/Baichuan2-7B-Base/",
    use_fast=False,
    trust_remote_code=True)
#可以打印出来看看
print(model)
print(tokenizer)
```
Model
```python
BaichuanForCausalLM(
  (model): BaichuanModel(
    (embed_tokens): Embedding(125696, 4096, padding_idx=0)
    (layers): ModuleList(
      (0-31): 32 x DecoderLayer(
        (self_attn): Attention(
          (W_pack): Linear(in_features=4096, out_features=12288, bias=False)
          (o_proj): Linear(in_features=4096, out_features=4096, bias=False)
          (rotary_emb): RotaryEmbedding()
        )
        (mlp): MLP(
          (gate_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (down_proj): Linear(in_features=11008, out_features=4096, bias=False)
          (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): RMSNorm()
        (post_attention_layernorm): RMSNorm()
      )
    )
    (norm): RMSNorm()
  )
  (lm_head): NormHead()
)
```

Tokenizer
```python
BaichuanTokenizer(name_or_path='./Baichuan2-7B-Base/', vocab_size=125696, model_max_length=4096, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'bos_token': '<s>', 'eos_token': '</s>', 'unk_token': '<unk>', 'pad_token': '<unk>'}, clean_up_tokenization_spaces=False, added_tokens_decoder={
	0: AddedToken("<unk>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	1: AddedToken("<s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
	2: AddedToken("</s>", rstrip=False, lstrip=False, single_word=False, normalized=True, special=True),
}
)
```

## 3. 训练模型
使用Trainer训练
```python
# 导入必要的库
from datasets import load_dataset  # 用于加载和处理数据集
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling  # 导入transformers相关组件
from peft import LoraConfig, get_peft_model , TaskTpye  # 导入LoRA相关组件，用于参数高效微调

##配置超参数
OUTPUT_DIR = "./lora_legal_qa_adapter" # 保存适配器的目录路径
SAVE_STEPS = 500 # 每多少步保存一次模型
EVAL_STEPS = 500 # 每多少步进行一次评估
tokenizer_max_length = 1024 # tokenizer的最大长度限制

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/" , use_fast=False , trust_remote_code=True)
tkenizer.pad_token = tokenizer.eos_token # 将pad_token设置为eos_token，确保有填充token

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
        inputs = f"\nhuman: {examplse["instruction"][i].strip()} {examples["inputs"][i].strip()} \n\nAssistant:"
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
        if len(input_ids) > tokenizer_max_length:
            input_id = input_id[:tokenizer_max_length]
            attention_mask = attention_mask[:tokenizer_max_length]
            label = label[:tokenizer_max_length]
        else:
            # 处理长度不足的情况：填充
            padding_length = tokenizer_max_length - len(input_id)
            input_id.extend([tokenizer.pad_token_id] * padding_length) # 用pad_token_id填充
            attention_mask.extend([0] * padding_length) # 注意力掩码填充部分设为0
            label.extend([-100] * padding_length) # 标签填充部分设为-100（忽略）
    
        # 将处理后的数据添加到批次中
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
        labels.append(label)

    # 返回处理后的批次数据
    return{
        "input_ids"： input_ids, # 输入ID序列
        "attention_masks"： attention_masks, # 注意力掩码
        "labels"： labels # 训练标签
    }

# 加载数据集
dataset = load_dataset("json", data_files="LoRA_data.jsonl" , split="train")
# 对数据集应用处理函数，批处理大小为4，移除原始列
dataset = dataset.map(process_function, batched=True , remove_columns=dataset.column_names , batch_size = 4)

# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1 , seed = 42) # 90%训练，10%验证，固定随机种子
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
    r = 8, # LoRA秩，控制适配器的大小
    lora_alpha = 32, # LoRA缩放参数
    lora_dropout = 0.1, # LoRA层的dropout率
    target_modules = ["o_proj", "gate_proj", "down_proj"] # 应用LoRA的目标模块
)

# 将模型转换为LoRA模型
model = get_peft_model(model , lora_config)
# 打印可训练参数信息
model.print_trainable_parameters()

# 配置训练参数
training_args = TrainingArguments(
    output_dir = OUTPUT_DIR, # 输出目录
    overwrite_output_dir = True, # 覆盖输出目录中的内容
    num_train_epochs = 3, # 训练轮数
    per_device_train_batch_size = 4, # 每个设备的训练批次大小
    per_device_eval_batch_size = 10, # 每个设备的评估批次大小
    learning_rate = 1e-5, # 学习率
    gradient_accumulation_steps = 10, # 梯度累积步数（模拟更大的批次大小）
    fp16 = True, # 使用混合精度训练
    logging_steps = 10, # 每多少步记录一次日志
    save_steps = SAVE_STEPS, # 保存间隔步数
    save_strategy = "steps", # 按步数保存策略
    eval_steps = EVAL_STEPS, # 评估间隔步数
    eval_strategy = "steps", # 按步数评估策略
    load_best_model_at_end = True, # 训练结束时加载最佳模型
    metric_for_best_model = "eval_loss", # 用于选择最佳模型的指标
    greater_is_better = False, # eval_loss越小越好
    warmup_steps = 100, # 学习率预热步数
    max_grad_norm = 1.0, # 梯度裁剪的最大范数
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
```
终端输入 python train.py 训练后可获得LoRA微调模型，但训练时间长，单卡A800-80G 约15H , 消耗显存大，约 55 ~ 60GB ,自己不想跑可以在我这里下
**[LoRA微调模型](https://huggingface.co/erfsdfds/BaiChuan2-7B-Law-SFT)**
下载完后解压，放在./lora_legal_qa_adapter路径下


## 4.测评部分
指标为困惑度，分别对Base和LoRA模型测评，并计算改进

**具体可看[CSDN博客-困惑度](https://blog.csdn.net/u013172930/article/details/145428394?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522191b3b52ecfe8257154d1774e06333b3%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=191b3b52ecfe8257154d1774e06333b3&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_click~default-1-145428394-null-null.142^v102^pc_search_result_base1&utm_term=%E5%9B%B0%E6%83%91%E5%BA%A6&spm=1018.2226.3001.4187)**

```python
import logging
import sys
from evaluate import load
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import math
from peft import PeftModel

# 设置logger
def setup_logger():
    """设置简单的logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('perplexity_evaluation.log', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

MAX_LENGTH = 1024

def calculate_perplexity_conservative(dataset, batch_size, device, model, tokenizer, optimized=False, max_samples=None):
    """
    更保守的困惑度计算方法
    """
    model_name = "Base" if not optimized else "LoRA"
    logger.info(f"开始计算 {model_name} 模型的困惑度")
    
    # 限制样本数量用于测试
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"限制样本数量为: {len(dataset)}")
    
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    total_loss = 0
    total_tokens = 0
    
    logger.info(f"开始计算 {len(dataset_list)} 条文本的困惑度...")
    
    for i in tqdm(range(0, len(dataset_list), batch_size)):
        batch_examples = dataset_list[i:i + batch_size]
        input_ids = []
        attention_masks = []
        labels = []
        
        try:
            for example in batch_examples:
                # 构建prompt和response
                prompt = f"\nHuman: {example['instruction'].strip()} {example['input'].strip()}\n\nAssistant:"
                prompt_enc = tokenizer(prompt, add_special_tokens=False)
                response = example["output"].strip() + tokenizer.eos_token
                response_enc = tokenizer(response, add_special_tokens=False)
                
                # 合并input_ids和attention_mask
                input_id = prompt_enc["input_ids"] + response_enc["input_ids"]
                attention_mask = prompt_enc["attention_mask"] + response_enc["attention_mask"]
                label = [-100] * len(prompt_enc["input_ids"]) + response_enc["input_ids"]
                
                # 截断或填充
                if len(input_id) > MAX_LENGTH:
                    input_id = input_id[:MAX_LENGTH]
                    attention_mask = attention_mask[:MAX_LENGTH]
                    label = label[:MAX_LENGTH]
                    logger.debug(f"文本被截断，长度: {len(input_id)}")
                else:
                    padding_length = MAX_LENGTH - len(input_id)
                    input_id.extend([tokenizer.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
                    label.extend([-100] * padding_length)
                
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
            
            # 转换为张量并移动到设备
            input_ids = torch.tensor(input_ids, dtype=torch.long).to(device)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                
                if loss is not None and not torch.isnan(loss):
                    # 关键修正：只计算非-100的token（response部分）
                    batch_tokens = (labels != -100).sum().item()
                    total_loss += loss.item() * batch_tokens
                    total_tokens += batch_tokens
                
                # 记录进度
                if i % 200 == 0 and total_tokens > 0:
                    avg_loss = total_loss / total_tokens
                    avg_perplexity = math.exp(avg_loss)
                    logger.info(f"{model_name}模型到{i}步 - 困惑度: {avg_perplexity:.4f}, 损失: {avg_loss:.4f}, 已处理token数: {total_tokens}")
                    
        except Exception as e:
            logger.error(f"批处理 {i//batch_size + 1} 计算失败: {e}")
            continue
    
    if total_tokens == 0:
        logger.error("所有批处理的困惑度计算失败")
        return {
            "model_name": model_name,
            "mean_perplexity": float('nan'),
            "mean_loss": float('nan'),
            "total_tokens": 0,
            "num_samples": len(dataset)
        }
    
    # 计算整体平均loss和困惑度
    avg_loss = total_loss / total_tokens
    avg_perplexity = math.exp(avg_loss)
    
    results = {
        "model_name": model_name,
        "mean_perplexity": avg_perplexity,
        "mean_loss": avg_loss,
        "total_tokens": total_tokens,
        "num_samples": len(dataset)
    }
    
    logger.info(f"{model_name}模型最终结果: {results}")
    
    return results

# 主程序
def main():
    logger.info("开始加载数据集和模型")
    
    dataset = load_dataset("json", data_files="/root/autodl-tmp/SFT/LoRA/LoRA_data.jsonl", split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    logger.info(f"测试数据集大小: {len(test_dataset)}")
    
    tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/", use_fast=False, trust_remote_code=True)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基线模型
    logger.info("加载基线模型")
    base_model = AutoModelForCausalLM.from_pretrained(
        "./Baichuan2-7B-Base/", 
        torch_dtype=torch.float16, 
        device_map="auto", 
        trust_remote_code=True
    )
    base_model.eval()
    
    # 加载LoRA模型
    logger.info("加载LoRA模型")
    lora_model = AutoModelForCausalLM.from_pretrained(
        "./Baichuan2-7B-Base/",
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_model, "./lora_legal_qa_adapter")
    lora_model.eval()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"使用设备: {device}")
    
    # 计算困惑度
    logger.info("开始计算基线模型困惑度")
    baseline_results = calculate_perplexity_conservative(
        test_dataset, batch_size=2, device=device, 
        model=base_model, tokenizer=tokenizer, optimized=False
    )
    
    logger.info("开始计算LoRA模型困惑度")
    optimized_results = calculate_perplexity_conservative(
        test_dataset, batch_size=2, device=device, 
        model=lora_model, tokenizer=tokenizer, optimized=True
    )
    
    # 计算改进
    improvement = {
        "baseline": baseline_results,
        "optimized": optimized_results,
        "improvement": {
            "perplexity_reduction": baseline_results["mean_perplexity"] - optimized_results["mean_perplexity"],
            "perplexity_improvement_pct": (
                (baseline_results["mean_perplexity"] - optimized_results["mean_perplexity"]) / 
                baseline_results["mean_perplexity"] * 100
            ) if baseline_results["mean_perplexity"] > 0 else 0,
            "loss_reduction": baseline_results["mean_loss"] - optimized_results["mean_loss"],
            "loss_improvement_pct": (
                (baseline_results["mean_loss"] - optimized_results["mean_loss"]) / 
                baseline_results["mean_loss"] * 100
            ) if baseline_results["mean_loss"] > 0 else 0
        }
    }
    
    logger.info("最终比较结果:")
    for key, value in improvement.items():
        if isinstance(value, dict):
            logger.info(f"{key}:")
            for k, v in value.items():
                if isinstance(v, float):
                    logger.info(f"  {k}: {v:.4f}")
                else:
                    logger.info(f"  {k}: {v}")
        else:
            if isinstance(value, float):
                logger.info(f"{key}: {value:.4f}")
            else:
                logger.info(f"{key}: {value}")
    
    logger.info("评估完成")

if __name__ == "__main__":
    main()
```

依旧，终端输入： python eval.py

最后结果为
```log
2025-10-07 13:25:24,299 - INFO - baseline:
2025-10-07 13:25:24,300 - INFO -   model_name: Base
2025-10-07 13:25:24,300 - INFO -   mean_perplexity: 4.4633
2025-10-07 13:25:24,300 - INFO -   mean_loss: 1.4959
2025-10-07 13:25:24,300 - INFO -   total_tokens: 1292113
2025-10-07 13:25:24,300 - INFO -   num_samples: 7970
2025-10-07 13:25:24,301 - INFO - optimized:
2025-10-07 13:25:24,301 - INFO -   model_name: LoRA
2025-10-07 13:25:24,301 - INFO -   mean_perplexity: 3.7871
2025-10-07 13:25:24,301 - INFO -   mean_loss: 1.3316
2025-10-07 13:25:24,302 - INFO -   total_tokens: 1292113
2025-10-07 13:25:24,302 - INFO -   num_samples: 7970
2025-10-07 13:25:24,302 - INFO - improvement:
2025-10-07 13:25:24,302 - INFO -   perplexity_reduction: 0.6762
2025-10-07 13:25:24,302 - INFO -   perplexity_improvement_pct: 15.1501
2025-10-07 13:25:24,303 - INFO -   loss_reduction: 0.1643
2025-10-07 13:25:24,303 - INFO -   loss_improvement_pct: 10.9826
```

## 5. 模型推理
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/", use_fast=False , trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("./Baichuan2-7B-Base/", trust_remote_code=True, torch_dtype = torch.float16, device_map = "auto")

from peft import PeftModel, PeftConfig
model = PeftModel.from_pretrained(base_model, "./lora_legal_qa_adapter")
print(model)
```
```python
PeftModelForCausalLM(
  (base_model): LoraModel(
    (model): BaichuanForCausalLM(
      (model): BaichuanModel(
        (embed_tokens): Embedding(125696, 4096, padding_idx=0)
        (layers): ModuleList(
          (0-31): 32 x DecoderLayer(
            (self_attn): Attention(
              (W_pack): Linear(in_features=4096, out_features=12288, bias=False)
              (o_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (rotary_emb): RotaryEmbedding()
            )
            (mlp): MLP(
              (gate_proj): lora.Linear(
                (base_layer): Linear(in_features=4096, out_features=11008, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=4096, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=11008, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (down_proj): lora.Linear(
                (base_layer): Linear(in_features=11008, out_features=4096, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=11008, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=4096, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
                (lora_magnitude_vector): ModuleDict()
              )
              (up_proj): Linear(in_features=4096, out_features=11008, bias=False)
              (act_fn): SiLU()
            )
            (input_layernorm): RMSNorm()
            (post_attention_layernorm): RMSNorm()
          )
        )
        (norm): RMSNorm()
      )
      (lm_head): NormHead()
    )
  )
)
```

```python
# 测试生成效果
test_questions = [
    "违章停车会受到什么处罚？",
    "劳动合同法保护哪些权益？",
    "如何申请专利？"
]

for question in test_questions:
    prompt = f"问题：{question}\n答案："
    inputs = tokenizer(prompt , return_tensors = "pt")
    with torch.no_grad():
        outputs = moedl.generate(
            **inputs,
            max_new_tokens = 200,
            temperature = 0.7,
            do_sample = True,
            padding_token_id = tokenizer.eos_token_id,
            use_cache = False
        )
    response = tokenizer.decode(outputs[0] , skip_special_tokens = True)
    print(f"问题：{question}")
    print(f"回答：{response[len(prompt):]}")
    print("-" * 50)
```
```python
问题：违章停车会受到什么处罚？
回答：1、在机动车道停车，罚款50元，不扣分。2、在非机动车道停车，罚款200元，扣2分，并拖走。3、在消防通道或人行走道停车，罚款200元，扣2分，并拖走。4、在人行道停车，罚款200元，扣2分，并拖走。5、停车超过3分钟，罚款50元。6、在禁停路段停车，罚款200元，扣2分，并拖走。7、不按规定地点停放，罚款200元，扣2分，并拖走。8、临时停车，罚款100元，不扣分。9、占用非机动车道，罚款100元，不扣分。10、占用盲道，罚款500元，不扣分。11、占用消防通道，罚款1000元，不扣分。12、占用公交车道，罚款
--------------------------------------------------
问题：劳动合同法保护哪些权益？
回答：答案：1.劳动合同的订立、履行和终止。2.劳动合同的变更和解除。3.集体合同的签订和履行。4.劳务派遣单位的设立。5.非全日制用工。6.试用期、服务期和违约金。7.用人单位的规章制度。8.劳动报酬。9.社会保险和福利。10.劳动安全卫生。11.女职工和未成年工特殊保护。12.工作时间和休息休假。13.职业培训。14.职业指导。15.劳动监督和劳动争议处理。16.劳动保护。
--------------------------------------------------
问题：如何申请专利？
回答：专利申请分为发明、实用新型和外观设计三种类型，申请专利需要准备相应的材料，具体如下：（1）发明专利申请，请求书、说明书（必要时应当有附图）、权利要求书、摘要及其附图各一式两份；（2）实用新型专利申请，请求书、说明书、摘要及其附图各一式两份；（3）外观设计专利申请，请求书、图片或者照片一式两份。申请外观设计专利的，还可以提交照片。要求保护色彩的，还应当提交彩色图片或者照片一式两份。委托专利代理机构的，应提交委托书。当事人直接办理申请的，应提交其身份证明文件。申请发明专利的，申请文件应当包括：（1）请求书：包括发明名称、申请人和发明人姓名、申请地址、联系方式、联系人、邮编、职务等；（2）说明书：包括独立权利要求、从属权利要求和摘要及其摘要附图。实用新型专利申请文件应当包括：（1）请求
```

## 6.构建RAG系统
需要下载bge-large-zh-v1.5模型

```python
import json
import torch
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging
from typing import Optional, List, Mapping, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义 LangChain 兼容的 LLM 类
class BaichuanLLM(LLM):
    """自定义 Baichuan LLM 包装器"""
    
    model: Any
    tokenizer: Any
    pipeline: Any
    
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        # 创建 transformers pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # 使用 pipeline 生成文本
            result = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            return result[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"生成文本时发生错误: {e}")
            return f"生成回答时出错: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "Baichuan2-7B-LoRA-Legal"}
    
    @property
    def _llm_type(self) -> str:
        return "baichuan_legal_qa"

#导入修复问题
try:
    from langchain_communtiy.llms import HuggingFacePipeline
except ImportError:
    logger.warning("langchain_community 未安装，尝试从 langchain 导入")
    from langchain.llms import HuggingFacePipeline

#加载数据
def load_qa_data(file_path: str)-> List[Document]:
    documents = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num , line in enumerate(f , 1):
                try:
                    data = json.loads(line)
                    page_content = f"问题：{data['input']}\n 答案：{data["output"]}"
                    metadata = {
                        "id":data.get("id" , f"line_{line_num}"),
                        "source": "legal_qa",
                        "line_number": line_num
                    }
                    documents.append(Document(page_content = page_content , metadata = metadata))
                except json.JSONDecodeError as e:
                    logger.error(f"第{line_num}行数据解析错误：{e}}")
                    continue
                except KeyError as e:
                    logger.error(f"第{line_num}行数据缺少字段：{e}")
            
        logger.info(f"成功加载{len(documents)}条数据")
        return documents
    except FileNotFoundError:
        logger.error(f"文件{file_path}不存在")
        raise
    except Exception as e:
        logger.error(f"加载数据时发生错误：{e}")
        raise

#将文件分块
def create_chunks(documents: List[Document], chunk_size: int = 256, chunk_overlap: int = 50)-> List[Document]:
    """
    将文档分块
    
    Args:
        documents: Document列表
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        
    Returns:
        分块后的Document列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size, 
        chunk_overlap = chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function = len,
        is_separator_regex = False)
    chunk = text_splitter.split_documents(documents)
    logger.info(f"成功将{len(documents)}条数据分块为{len(chunk)}个块")
    return chunk

def create_vector_store(chunk: List[Document], model_path: str = "./bge-large-zh-v1.5/")-> FAISS:
    """
    创建向量存储
    
    Args:
        chunks: 文档块列表
        model_path: 嵌入模型路径
        
    Returns:
        FAISS向量存储
    """
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name = model_path,
            model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs = {"normalize_embeddings": True,
                            "batch_size": 64}
        )

        #创建向量数据库
        vector_store = FAISS.from_documents(chunk, embeddings)
        logger.info("成功创建向量数据库")
        return vector_store , embeddings
    except Exception as e:
        logger.error(f"创建向量数据库时发生错误：{e}")
        raise

#储存数据库到本地
def save_vector_store(vector_store: FAISS, file_path: str = "faiss_legal_qa_index")-> None:
    try:
        vector_store.save_local(file_path)
        logger.info(f"成功将向量数据库保存到{file_path}")
    except Exception as e:
        logger.error(f"保存向量数据库时发生错误：{e}")
        raise

#加载数据库
def load_vector_store(path: str, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    从本地加载向量存储
    
    Args:
        path: 加载路径
        embedding_model: 嵌入模型
        
    Returns:
        FAISS向量存储
    """
    try:
        vector_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"向量数据库已从 {path} 加载")
        return vector_db
    except Exception as e:
        logger.error(f"加载向量数据库时发生错误: {e}")
        raise

#搜索相似文档
def search_similar_documents(vector_store: FAISS, query: str, k: int = 10)-> List[Document]:
    """
    搜索相似文档
    
    Args:
        vector_db: FAISS向量存储
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        相似文档列表
    """
    try:
        similar_docs = vector_store.similarity_search(query, k=k)
        logger.info(f"查询 '{query}' 找到 {len(similar_docs)} 个相似文档")
        return similar_docs
    except Exception as e:
        logger.error(f"搜索相似文档时发生错误：{e}")
        raise

def initalize_model(base_model_path: str, lora_adapter_path: str)    """
    初始化模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA适配器路径
        
    Returns:
        模型和tokenizer
    """
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype = torch.float16,
            device_map = "auto",
            trust_remote_code = True,
        )
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)

        tokenizer = AutoTokenizer.from_pretrained(base_model_path , trust_remote_code = True)
        logger.info("成功初始化模型")
        return model, tokenizer
    except Exception as e:
        logger.error(f"初始化模型时发生错误：{e}")
        raise

def create_qa_chain(model , tokenizer , vector_store: FAISS , k: int = 3):
    """
    创建问答链
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        vector_db: 向量数据库
        k: 检索文档数量
        
    Returns:
        QA链
    """
    try:
        from langchain.llms import HuggingFacePipeline
        #创建Pipeline
        text_generation = pipeline(
            "text-generation",
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = 512,
            temperature = 0.7,
            top_p = 0.9,
            repeat_penalty = 1.1,
            pad_token_id = tokenizer.eos_token_id
        )
        #构建Langchain LLM
        llm = HuggingFacePipeline(pipeline = text_generation)

        qa_chain = RetrievalQA.from_chain_type(
            llm = llm,
            chain_type = "stuff",
            retriever = vector_store.as_retriever(search_kwargs = {"k": k}),
            return_source_documents = True
        )
        logger.info("成功创建问答链")
        return qa_chain
    except Exception as e:
        logger.error(f"创建问答链时发生错误：{e}")
        raise

def main():
    # 配置参数
    data_file_path = "/root/autodl-tmp/SFT/LoRA/LoRA_data.jsonl"
    embedding_model_path = "./bge-large-zh-v1.5/"
    base_model_path = "./Baichuan2-7B-Base/"
    lora_adapter_path = "./lora_legal_qa_adapter"
    faiss_index_path = "faiss_legal_qa_index"

    try:
        # 1.加载数据
        documents = load_qa_data(data_file_path)

        # 2.数据分块
        chunk = create_chunks(documents)

        # 3.创建向量数据库
        vector_store, embeddings = create_vector_store(chunk, embedding_model_path)

        # 4.保存数据库
        save_vector_store(vector_store, faiss_index_path)

        # 5.相似性搜索
        query = "劳动合同法保护哪些权益？"
        similar_docs = search_similar_documents(vector_store, query)

        print(f"针对查询：\n{query}")
        print("找到的相似文档：")
        for idx, doc in enumerate(similar_docs):
            print(f"{idx + 1}: {doc.page_content}")
            print(f"元数据：{doc.metadata}\n")

        # 6.初始化模型
        model , tokenizer = initalize_model(base_model_path, lora_adapter_path)

        # 7.创建问答链
        qa_chain = create_qa_chain(model, tokenizer, vector_store)

        # 8.问答
        result = qa_chain.invoke({"query": query})
        print(f"生成的答案：\n{result['result']}")
        
        # 显示源文档
        print("\n参考的源文档：")
        for idx, doc in enumerate(result['source_documents']):
            print(f"文档 {idx + 1}: {doc.page_content}")
    except Exception as e:
        logger.error(f"程序发生错误：{e}")
        raise
if __name__ == "__main__":
    main()
```
终端输入 python data_chain.py

大致生成如下：
```python
参考的源文档：
root@autodl-container-98b24ebe42-16fe1f96:~/autodl-tmp# python new_chain.py
INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: ./bge-large-zh-v1.5/
INFO:faiss.loader:Loading faiss with AVX512 support.
INFO:faiss.loader:Successfully loaded faiss with AVX512 support.
INFO:__main__:向量数据库已从 faiss_legal_qa_index 加载
INFO:__main__:查询 '劳动合同法保护哪些权益？' 找到 10 个相似文档
针对查询：
劳动合同法保护哪些权益？
找到的相似文档：
1: 问题：某公司未依法与职工签订劳动合同，又未参加社会保险，同时也没有提供充足的职业教育和岗位培训，该如何保护职工的合法权益？
答案：根据《公司法》第十七条和第十八条的规定，针对某公司未依法与职工签订劳动合同、未参加社会保险，以及未提供充足的职业教育和岗位培训等情况，可以采取以下措施来保护职工的合法权益：

1. 公司应当依法与职工签订劳动合同，确保职工合法权益的得到尊重和保护。
2. 公司必须参加社会保险，确保职工享有相应的社会保险权益，包括养老保险、医疗保险、失业保险等。
3. 公司应当加强劳动保护，确保职工的劳动安全和健康。
4. 公司应该采取多种形式，加强公司职工的职业教育和岗位培训，提高职工素质。
5. 公司职工可以依照《中华人民共和国工会法》的规定自行组织工会，通过工会来维护职工的合法权益。
6. 公司应当为本公司工会提供必要的活动条件，支持工会的正常运作。
7. 公司工会可以代表职工与公司签订集体合同，就职工的劳动报酬、工作时间、福利、保险和劳动安全卫生等事项进行协商和维护。
8. 公司在重大决策和规章制度制定过程中，应当听取公司工会的意见，并适时听取职工的意见和建议。

总之，以上措施可以帮助保护职工的合法权益，确保公司依法履行对职工的义务，促进公平就业和劳动关系稳定。
元数据：{'id': 'line_31027', 'source': 'legal_qa', 'line_number': 31027}

2: 问题：某企业的劳动者因为超时工作导致身体出现不适，在向企业管理层提出要求停止加班时，却遭到了拒绝。劳动者想知道有哪些法律可以保护他们的权益。

劳动者有哪些权利可以保障他们的劳动权益？
答案：根据所提供的相关法律条文，《劳动法》确保劳动者的劳动权益。以下是劳动者的一些权利：

1. 平等就业和选择职业的权利（《劳动法》第三条）：劳动者有权公平地从事职业，并自主选择自己的工作。

2. 取得劳动报酬的权利（《劳动法》第三条）：劳动者有权获得与自己劳动成果相符的合理报酬。

3. 休息休假的权利（《劳动法》第三条）：劳动者有享受休息休假的权利，包括法定节假日和带薪年假。

4. 获得劳动安全卫生保护的权利（《劳动法》第三条）：劳动者有权在工作场所获得良好的劳动安全环境和必要的劳动保护设施。

5. 接受职业技能培训的权利（《劳动法》第三条）：劳动者有权接受与工作相关的职业技能培训，提升自己的工作能力。

6. 享受社会保险和福利的权利（《劳动法》第三条）：劳动者有权享受社会保险和福利，包括养老保险、医疗保险、失业保险和工伤保险等。

7. 提请劳动争议处理的权利（《劳动法》第三条）：劳动者有权依法提起劳动争议，并通过合法程序解决劳动纠纷。

总之，劳动者在工作中享有合法权益，包括就业机会选择权、合理报酬权、休息权、安全保护权、培训权、社会保险权、劳动争议处理权等。用人单位应建立和完善规章制度，保障劳动者行使这些权利，并共同努力提高劳动者的生活水平。（《劳动法》第四条、第五条）
元数据：{'id': 'line_29706', 'source': 'legal_qa', 'line_number': 29706}

3: 问题：劳动合同应具备哪些条款？
答案：《劳动合同法》对劳动合同必备条款的规定包括这样几个方面：?用人单位的基本情况：如名称、住所和法定代表人或者主要负责人?劳动者的主要情况：如姓名、住址、居民身份证或者其他有效身份证件号码?劳动合同期限?工作内容和工作地点?工作时间和休息休假?劳动报酬?社会保险?劳动保护、劳动条件和职业危害防护?法律法规规定应当纳入劳动合同的其他事项此外，劳动者和用人单位可以约定试用期、培训、保密、补充保险和福利待遇等其他事项
元数据：{'id': 'line_5003', 'source': 'legal_qa', 'line_number': 5003}

4: 问题：某家企业未依法保障员工权益，是否违反了《社会法-就业促进法》？有哪些组织可以协助维护员工权益？
答案：根据《就业促进法》第八条的规定，用人单位应当依法保障劳动者的合法权益。如果某家企业未依法保障员工权益，可以认定违反了《社会法-就业促进法》。

针对这种情况，可以有一些组织可以协助维护员工权益。根据《就业促进法》第九条的规定，工会、共产主义青年团、妇女联合会、残疾人联合会以及其他社会组织都可以依法维护劳动者的劳动权利，它们可以协助人民政府开展促进就业工作，并提供维护员工权益的支持。

因此，如果某家企业未依法保障员工权益，员工可以寻求工会、共产主义青年团、妇女联合会、残疾人联合会或其他社会组织的帮助来维护自己的权益。同时，县级以上人民政府和有关部门也有责任统筹协调产业政策与就业政策，如果相关组织无法维护员工权益，员工还可以向人民政府和有关部门反映情况，寻求进一步的支持和保护。

显示成效的单位和个人可以根据《就业促进法》第十条获得表彰和奖励，而国家鼓励各类企业增加就业岗位。根据《就业促进法》第十二条，国家鼓励各类企业通过兴办产业或者拓展经营增加就业岗位，同时扶持中小企业和发展劳动密集型产业、服务业，以扩大就业。国家也鼓励、支持、引导非公有制经济发展，增加就业岗位。

请注意，以上回答是根据您提供的法律条文和事实得出的结论。
元数据：{'id': 'line_34053', 'source': 'legal_qa', 'line_number': 34053}

5: 问题：某公司的员工发现自己的工资和劳动合同不符，他们希望通过工会维护自己的合法权益。根据《中华人民共和国工会法》，工会有哪些权利和义务？
答案：根据《中华人民共和国工会法》，工会享有以下权利和义务：

1. 权利：
   - 维护和捍卫劳动者的合法权益，包括工资、劳动条件、职业安全与健康等方面的权益。
   - 参与和监督企业制定和完善劳动规章制度。
   - 参与劳动关系协调与调解，维护劳动者与雇主之间的合法权益。
   - 协助劳动争议的解决与调解，包括组织和参与劳动仲裁、劳动法庭的程序。
   - 参与劳动保护监督，监督雇主的合法用工和保护职工的权益。
   - 开展职工教育培训，提高劳动者的技能水平和工作能力。
   - 参与企业决策与管理，维护职工合法权益的代表。

2. 义务：
   - 组织企业员工参加工会，保障劳动者加入工会的自由和平等权利。
   - 代表职工与雇主协商和签订劳动合同，维护劳动者的合法权益。
   - 组织劳动者参与工会活动，培育和发展工会组织。
   - 向工会会员提供法律援助，维护劳动者的合法权益。
   - 履行国家委托的其他职责，如协助实施劳动法律法规，参与劳动争议的调解等。

需要注意的是，以上权利和义务应在法律规定的范围内行使和履行。如有具体情况需要更详细的法律分析和解释，请咨询专业法律机构或律师。
元数据：{'id': 'line_25491', 'source': 'legal_qa', 'line_number': 25491}

6: 问题：某公司采取不公平的劳动制度，不给员工提供合理的薪资、休息时间和健康安全保障等，导致员工的合法权益受到侵害。在此情况下，员工可以通过什么途径来维护自己的权益？
答案：根据《工会法》第四条、第五条和第六条，员工可以通过工会来维护自己的权益。工会的基本职责是维护职工的合法权益，并通过平等协商和集体合同制度等方式，推动健全劳动关系协调机制，维护职工劳动权益，构建和谐劳动关系。工会通过职工代表大会或其他形式组织职工参与本单位的民主选举、民主协商、民主决策、民主管理和民主监督。工会还建立联系广泛、服务职工的工会工作体系，听取和反映职工的意见和要求，关心职工的生活，帮助职工解决困难，全心全意为职工服务。

所以，员工可以通过参加工会并积极参与工会活动来维护自己的权益。工会可以代表员工与公司进行协商，促使公司提供合理的薪资、休息时间和健康安全保障。员工还可以借助工会的组织力量，通过集体行动、集体谈判等方式来维护自己的权益。重要的是，员工应该了解和行使自己的民主权利，积极参与工会的活动和决策过程，以实现自己的合法权益保护。
元数据：{'id': 'line_17194', 'source': 'legal_qa', 'line_number': 17194}

7: 问题：某工厂的工会收到了员工关于工作安全问题的反映。工会决定组织员工代表与厂方进行协商，并要求在谈判中维护员工的合法权益和劳动权益。

工会在谈判中要求维护员工的哪些合法权益和劳动权益？
答案：根据《工会法》第六条，工会在维护职工合法权益时有以下职责：
1. 维护职工的劳动权益：工会通过平等协商和集体合同制度等方式，推动健全劳动关系协调机制，维护职工的劳动权益。
2. 代表和维护职工的合法权益：工会代表职工维护其合法权益，包括但不限于工资待遇、工作条件、工作安全等方面的权益。
3. 构建和谐劳动关系：工会通过参与民主选举、民主协商、民主决策、民主管理和民主监督等方式，维护职工的劳动权益，构建和谐的劳动关系。

另根据《工会法》第七条，工会也有动员和组织职工积极参加经济建设的职责，提高职工的思想道德、技术业务和科学文化素质，建设有理想、有道德、有文化、有纪律的职工队伍，这也是维护职工的劳动权益的一种方式。

因此，在谈判中，工会应维护员工的合法权益和劳动权益，包括但不限于工资待遇、工作条件、工作安全等方面的权益，同时也有义务促进职工的思想道德提高和建设有素质的职工队伍。
元数据：{'id': 'line_27897', 'source': 'legal_qa', 'line_number': 27897}

8: 问题：劳动合同的规定有哪些?
答案：《劳动合同法》第十条 【订立书面劳动合同】建立劳动关系，应当订立书面劳动合同。已建立劳动关系，未同时订立书面劳动合同的，应当自用工之日起一个月内订立书面劳动合同。用人单位与劳动者在用工前订立劳动合同的，劳动关系自用工之日起建立。《劳动合同法》第十一条 【未订立书面劳动合同时劳动报酬不明确的解决】用人单位未在用工的同时订立书面劳动合同，与劳动者约定的劳动报酬不明确的，新招用的劳动者的劳动报酬按照集体合同规定的标准执行；没有集体合同或者集体合同未规定的，实行同工同酬。第十二条 【劳动合同的种类】劳动合同分为固定期限劳动合同、无固定期限劳动合同和以完成一定工作任务为期限的劳动合同。
元数据：{'id': 'line_12718', 'source': 'legal_qa', 'line_number': 12718}

9: 问题：如果一个企业侵害了劳动者的合法权益，造成了财产损失或其他损害，可能会面临哪些责任？
答案：根据《就业促进法》第六十八条规定，如果一个企业侵害了劳动者的合法权益，造成财产损失或其他损害，该企业将承担民事责任，依法进行赔偿。另外，如果这种侵害构成犯罪，还将依法追究刑事责任。

根据《就业促进法》第六十九条规定，该法自2008年1月1日起施行。

根据《退役军人保障法》第一条规定，为了加强退役军人保障工作，维护退役军人合法权益，根据宪法，制定了该法。

根据《退役军人保障法》第二条规定，退役军人是指从中国人民解放军依法退出现役的军官、军士和义务兵等人员。

综上所述，如果一个企业侵害了劳动者的合法权益，造成了财产损失或其他损害，该企业应依法承担民事责任并赔偿受损劳动者。
元数据：{'id': 'line_26712', 'source': 'legal_qa', 'line_number': 26712}

10: 问题：某公司女员工因怀孕被公司解除劳动合同，她可以依据哪些法律规定维护自身合法权益？
答案：根据提供的相关法律规定，《妇女权益保障法》的第三条规定，各级人民政府应该重视和加强妇女权益的保障工作，并负责组织、协调、指导、督促有关部门做好妇女权益的保障工作。根据该法第四条，保障妇女的合法权益是全社会的共同责任，国家机关、社会团体、企业事业单位等都须依法保障妇女的权益。此外，根据该法第七条，国家鼓励妇女运用法律维护自身合法权益。

因此，某公司女员工可以根据《妇女权益保障法》的相关规定维护自身合法权益。她可以向有关部门投诉或寻求法律援助，要求公司依法保障其权益，包括保留劳动合同、禁止性别歧视、提供必要的工作调整等。
元数据：{'id': 'line_35236', 'source': 'legal_qa', 'line_number': 35236}

`torch_dtype` is deprecated! Use `dtype` instead!
WARNING:transformers_modules.modeling_baichuan:Xformers is not installed correctly. If you want to use memory_efficient_attention to accelerate training use the following command to install Xformers
pip install xformers.
INFO:accelerate.utils.modeling:We will use 90% of the memory on device 0 for storing the model, and 10% for the buffer to avoid OOM. You can set `max_memory` in to a higher value to use more memory (at your own risk).
Loading checkpoint shards: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:02<00:00,  1.45s/it]
INFO:__main__:模型加载成功
使用简单 RAG 方法进行问答...
/root/miniconda3/lib/python3.12/contextlib.py:105: FutureWarning: `torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature.
  self.gen = func(*args, **kwds)
生成的答案：
根据法律规定，《劳动合同法》旨在保护劳动者和用人单位双方的合法权益，从而构建和谐稳定的劳动关系。具体来说，该法保护如下几项权益：

1. 订立劳动合同的自由及自愿原则：双方都有权决定是否签订劳动合同，并有选择适合自己的用工形式或工种的权利。

2. 劳动者的知情权和监督权：劳动者有权了解用人单位的基本情况和劳动报酬标准；也有权参与民主管理，并对用人单位的规章制度提出建议。

3. 劳动纪律和规章制度：用人单位依法制定的规章制度须符合国家法律法规，且内容明确具体、便于操作执行。

4. 女职工和未成年工的特殊保护：用人单位不得安排女职工从事禁忌作业，也不得在怀孕期、哺乳期解除劳动合同或降低其工资标准。此外，用人单位还应对未成年工实行特殊保护。

5. 社会保险和福利：用人单位必须为劳动者缴纳社会保险，并提供符合法定标准的福利待遇。

6. 劳动保护：用人单位要为劳动者提供符合国家规定的劳动安全卫生条件，并依法为劳动者提供必要的劳动防护用品和安全设施。

7. 经济补偿金和赔偿金：当劳动合同因违反法律而解除或终止时，用人单位需要支付经济补偿金给劳动者，若给劳动者造成损害的还需承担赔偿责任。

8. 劳务派遣单位的选择：劳动者可以与劳务派遣单位订立劳务协议，并由后者派遣到用工单位从事相应的工作。但是，劳务派遣单位应当具备相应资质和条件，并且按照合同约定向被派遣劳动者支付劳动报酬和相关待遇。

9. 劳务派遣中三方的关系和责任：由于劳务派遣涉及三方关系，因此有必要明晰三方的责任范围及其相互间的法律责任和义务。例如，如果发生争议或纠纷，应由谁作为主体提起诉讼或仲裁等问题都需要进一步明确界定。

综上所述，《劳动合同法》致力于保护劳动者的合法权益，使其能够安心工作并获得应有的回报和保障。同时也要求用人单位遵守相关法律法规，确保劳动关系和谐稳定发展。

参考的源文档：
文档 1: 问题：某公司未依法与职工签订劳动合同，又未参加社会保险，同时也没有提供充足的职业教育和岗位培训，该如何保护职工的合法权益？
答案：根据《公司法》第十七条和第十八条的规定，针对某公司未依法与职工签订劳动合同、未参加社会保险，以及未提供充足的职业教育和岗位培训等情况，可以采取以下措施来保护职工的合法权益：

1. 公司应当依法与职工签订劳动合同，确保职工合法权益的得到尊重和保护。
2. 公司必须参加社会保险，确保职工享有相应的社会保险权益，包括养老保险、医疗保险、失业保险等。
3. 公司应当加强劳动保护，确保职工的劳动安全和健康。
4. 公司应该采取多种形式，加强公司职工的职业教育和岗位培训，提高职工素质。
5. 公司职工可以依照《中华人民共和国工会法》的规定自行组织工会，通过工会来维护职工的合法权益。
6. 公司应当为本公司工会提供必要的活动条件，支持工会的正常运作。
7. 公司工会可以代表职工与公司签订集体合同，就职工的劳动报酬、工作时间、福利、保险和劳动安全卫生等事项进行协商和维护。
8. 公司在重大决策和规章制度制定过程中，应当听取公司工会的意见，并适时听取职工的意见和建议。

总之，以上措施可以帮助保护职工的合法权益，确保公司依法履行对职工的义务，促进公平就业和劳动关系稳定。
文档 2: 问题：某企业的劳动者因为超时工作导致身体出现不适，在向企业管理层提出要求停止加班时，却遭到了拒绝。劳动者想知道有哪些法律可以保护他们的权益。

劳动者有哪些权利可以保障他们的劳动权益？
答案：根据所提供的相关法律条文，《劳动法》确保劳动者的劳动权益。以下是劳动者的一些权利：

1. 平等就业和选择职业的权利（《劳动法》第三条）：劳动者有权公平地从事职业，并自主选择自己的工作。

2. 取得劳动报酬的权利（《劳动法》第三条）：劳动者有权获得与自己劳动成果相符的合理报酬。

3. 休息休假的权利（《劳动法》第三条）：劳动者有享受休息休假的权利，包括法定节假日和带薪年假。

4. 获得劳动安全卫生保护的权利（《劳动法》第三条）：劳动者有权在工作场所获得良好的劳动安全环境和必要的劳动保护设施。

5. 接受职业技能培训的权利（《劳动法》第三条）：劳动者有权接受与工作相关的职业技能培训，提升自己的工作能力。

6. 享受社会保险和福利的权利（《劳动法》第三条）：劳动者有权享受社会保险和福利，包括养老保险、医疗保险、失业保险和工伤保险等。

7. 提请劳动争议处理的权利（《劳动法》第三条）：劳动者有权依法提起劳动争议，并通过合法程序解决劳动纠纷。

总之，劳动者在工作中享有合法权益，包括就业机会选择权、合理报酬权、休息权、安全保护权、培训权、社会保险权、劳动争议处理权等。用人单位应建立和完善规章制度，保障劳动者行使这些权利，并共同努力提高劳动者的生活水平。（《劳动法》第四条、第五条）
文档 3: 问题：劳动合同应具备哪些条款？
答案：《劳动合同法》对劳动合同必备条款的规定包括这样几个方面：?用人单位的基本情况：如名称、住所和法定代表人或者主要负责人?劳动者的主要情况：如姓名、住址、居民身份证或者其他有效身份证件号
```

## 优化
### 2025.10.24
一轮LoRA训练后，回答还是不够好，进行第二次LoRA训练，所选数据集为**[法律QA](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT), 其中取 DISC-Law-SFT-Triplet-QA-released.jsonl 这一项**
下载完后放在当前文件夹后解压

```python
from datasets import load_dataset
file_path = "/root/autodl-tmp/SFT/Triplet/DISC-Law-SFT-Triplet-QA-released.jsonl" #这里用你自己的
data = load_dataset("jsnn",  data_files = file_path , split = "train")
print(data[0])
```

结果如下

```json
{'id': 'legal_question_answering_1',
 'reference': ['《民法典》第九百七十三条：合伙人对合伙债务承担连带责任。清偿合伙债务超过自己应当承担份额的合伙人，有权向其他合伙人追偿。',
  '《民法典》第九百七十四条：除合伙合同另有约定外，合伙人向合伙人以外的人转让其全部或者部分财产份额的，须经其他合伙人一致同意。',
  '《民法典》第九百七十五条：合伙人的债权人不得代位行使合伙人依照本章规定和合伙合同享有的权利，但是合伙人享有的利益分配请求权除外。',
  '《民法典》第九百七十六条：合伙人对合伙期限没有约定或者约定不明确，依据本法第五百一十条的规定仍不能确定的，视为不定期合伙。合伙期限届满，合伙人继续执行合伙事务，其他合伙人没有提出异议的，原合伙合同继续有效，但是合伙期限为不定期。合伙人可以随时解除不定期合伙合同，但是应当在合理期限之前通知其他合伙人。'],
 'input': '《民法典》第九百七十四条：除合伙合同另有约定外，合伙人向合伙人以外的人转让其全部或者部分财产份额的，须经其他合伙人一致同意。\n《民法典》第九百七十五条：合伙人的债权人不得代位行使合伙人依照本章规定和合伙合同享有的权利，但是合伙人享有的利益分配请求权除外。\n《民法典》第九百七十三条：合伙人对合伙债务承担连带责任。清偿合伙债务超过自己应当承担份额的合伙人，有权向其他合伙人追偿。\n《民法典》第九百七十六条：合伙人对合伙期限没有约定或者约定不明确，依据本法第五百一十条的规定仍不能确定的，视为不定期合伙。合伙期限届满，合伙人继续执行合伙事务，其他合伙人没有提出异议的，原合伙合同继续有效，但是合伙期限为不定期。合伙人可以随时解除不定期合伙合同，但是应当在合理期限之前通知其他合伙人。\n<问题>：\n在不定期合伙的情况下，一个合伙人突然退出并清偿了其应当承担份额的合伙债务后，是否有权向其他合伙人追偿？',
 'output': '根据《民法典》第九百七十三条规定，合伙人对合伙债务承担连带责任。当一个合伙人突然退出并清偿了其应当承担份额的合伙债务后，依法有权向其他合伙人追偿。'}
```

这里可以看出在input里加入了参考文档，因此回答效果会较第一轮LoRA训练要好

进行训练

```python
# train_lora_from_new_sft.py
import os
import math
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

# ----------------- 配置区（按需改） -----------------
BASE_MODEL = "./Baichuan2-7B-Base/"          # 你的 base 模型目录
LORA_MODEL = ""
SFT_JSONL  = "/root/autodl-tmp/SFT/Triplet/DISC-Law-SFT-Triplet-QA-released.jsonl"  # 新 SFT 数据 (jsonl)
OUT_DIR    = "./lora_new_sft_adapter"         # LoRA 输出目录
MAX_LENGTH = 1024
EPOCHS     = 3
BATCH_SIZE = 4           # 若 OOM -> 降到 1
GRAD_ACCUM = 1
LR         = 1e-5
TARGET_MODULES = ["o_proj", "gate_proj", "down_proj"]  # 按你上次设置
SAVE_STEPS = 250           # 保存步数（示例）
EVAL_STEPS = 250           # 评估步数，必须与 save 策略/间隔一致以使用 load_best_model_at_end
# --------------------------------------------------

# 1) 加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False, trust_remote_code=True)
# 确保有 pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 2) 加载原始 jsonl（每行 dict，包含 id, reference(list), input, output）
raw = load_dataset("json", data_files={"train": SFT_JSONL}, split="train")
print("raw sample count:", len(raw))

# 3) 将原始样本转为 SFT prompt/response 格式
#    我这里采用：prompt 包含已知法律条文(reference) + 问题(input)
#    response 就是 output（你的 ground-truth 回答）
def make_prompt(example):
    # 把 reference list 拼到一段（若为空则忽略）
    refs = example.get("reference", None)
    if isinstance(refs, list):
        refs_text = "\n".join([r.strip() for r in refs if r is not None and len(r.strip())>0])
    elif refs is None:
        refs_text = ""
    else:
        refs_text = str(refs).strip()

    # 构造 prompt（你之前训练时使用了 "Human/Assistant" 模式）
    prompt_parts = []
    if refs_text:
        prompt_parts.append(f"已知条文：\n{refs_text}")
    prompt_parts.append("请基于以上条文，回答下列问题：")
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
print("示例转换完成，前2条：")
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
        if len(input_ids) > MAX_LENGTH:
            # 保留后 MAX_LENGTH tokens（以尽可能保留response）
            input_ids = input_ids[-MAX_LENGTH:]
            attention_mask = attention_mask[-MAX_LENGTH:]
            labels = labels[-MAX_LENGTH:]
        else:
            # 不在这里 pad（交给 data_collator 做 dynamic padding），但为了避免 extremely short example 我们可以不 pad
            padding_length = MAX_LENGTH - len(input_ids)
            input_ids.extend([tokenizer.pad_token_id] * padding_length)
            attention_mask.extend([0] * padding_length)
            labels.extend([-100] * padding_length)
            

        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(labels)

    return {"input_ids": input_ids_list, "attention_mask": attention_mask_list, "labels": labels_list}

tokenized = pairwise.map(tokenize_and_build_labels, batched=True, remove_columns=pairwise.column_names , batch_size = 4)
print("tokenized example count:", len(tokenized))
print("示例 tokenized（第0条）长度：", len(tokenized[0]["input_ids"]))

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
    BASE_MODEL,
    trust_remote_code=True,
    dtype=torch.float16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./lora_legal_qa_adapter")

# 确保 LoRA 层启用梯度计算
for param in model.parameters():
    param.requires_grad = True

# LoRA 配置
lora_config = LoraConfig(
    task_type="CAUSAL_LM",        # 任务类型
    inference_mode=False,         # 训练模式
    r=8,                          # LoRA秩
    lora_alpha=32,                # LoRA缩放系数
    lora_dropout=0.1,             # Dropout
    target_modules=["o_proj", "gate_proj", "down_proj"]  # LoRA适配层
)

# 再次加载 LoRA 配置
model = get_peft_model(base_model, lora_config)

# 打印可训练参数
model.print_trainable_parameters()

# 8) TrainingArguments + Trainer
training_args = TrainingArguments(
    output_dir=OUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=10,
    learning_rate=1e-5,
    fp16=True,
    logging_steps=50,
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
os.makedirs(OUT_DIR, exist_ok=True)
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("LoRA adapter saved to:", OUT_DIR)
```

最后结果
```python
{
  "epoch": 2.9523809523809526,
  "grad_norm": 14918.9248046875,
  "learning_rate": 1.76271186440678e-07,
  "loss": 0.4775,
  "step": 1550
}
```

这里的Loss比第一次训练要低了,最后用langchain构建

```python
import os
import json
import logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# LangChain subpackages (these should work in your env)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
```

## 7.想法与改进
### 7.1 可视化界面
### 7.2 微调后生成的模型对话生硬，可在基础上进行DPO优化
### 7.3 QA数据未进行清洗与筛选，如困惑度筛选，去重等
### 7.4 未成功尝试多卡训练，之前的3卡4090没跑成
### 7.5 LangChain流程过于简单，需要进行优化
### 7.6 刚学了两个月LLM就来做东西，有些东西感觉没说明白

## 8.更新
### 2025.10.22
训了第二轮LoRA,明天补上

### 2025.10.18
比样的 SFT数据我给用来RL了，气笑了


















