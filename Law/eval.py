def calculate_perplexity_conservative(dataset, batch_size , device , model , tokenizer, optimized=False):
    """
    更保守的困惑度计算方法
    """
    model_name = "Base" if not optimized else "LoRA"
    print(model)
    dataset_list = [dataset[i] for i in range(len(dataset))]
    
    total_loss = 0
    total_tokens = 0
    
    print(f"开始计算 {len(dataset_list)} 条文本的困惑度...")
    
    for i in tqdm(range(0, 25, batch_size)):
        batch_examples = dataset_list[i:i + batch_size]
        input_ids = []
        attention_masks = []
        labels = []
        
        try:
            texts = []
            for example in batch_examples:
         
                prompt = f"\n Human: {example["instruction"].strip()} {example["input"].strip()} \n\n Assistant:"
                prompt = tokenizer(prompt , add_special_tokens = False)
                response = example["output"].strip() + tokenizer.eos_token
                response = tokenizer(response , add_special_tokens = False)
                input_id = prompt["input_ids"] + response["input_ids"]
                attention_mask = prompt["attention_mask"] + response["attention_mask"]
                label = [-100] * len(prompt["input_ids"]) + response["input_ids"]
    
                if len(input_id) > MAX_LENGTH:
                    input_id = input_id[:MAX_LENGTH]
                    attention_mask = attention_mask[:MAX_LENGTH]
                    label = label[:MAX_LENGTH]
                else:
                    padding_length = MAX_LENGTH - len(input_id)
                    input_id.extend([tokenizer.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)
                    label.extend([-100] * padding_length)
                input_ids.append(input_id)
                attention_masks.append(attention_mask)
                labels.append(label)
    
            # 转换为张量
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_masks = torch.tensor(attention_masks, dtype=torch.long)
            labels = torch.tensor(labels, dtype=torch.long)
            inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_masks,
                        "labels": labels
                    }
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                loss = outputs.loss
                
                if loss is not None and not torch.isnan(loss):
                    # 累计总loss和总token数
                    batch_tokens = inputs["attention_mask"].sum().item()
                    total_loss += loss.item() * batch_tokens
                    total_tokens += batch_tokens
                if i % 10 == 0:
                    avg_loss = total_loss / total_tokens
                    avg_perplexity = math.exp(avg_loss)
                    print(f"{model_name}模型到{i}步的困惑值")
                    results = {
                        "model_name": model_name,
                        "mean_perplexity": avg_perplexity,
                        "mean_loss": avg_loss,
                        "total_tokens": total_tokens,
                        "num_samples": len(dataset)
                    }
                    print(results)
                    print("=========\n")
                    
                    
        except Exception as e:
            print(f"批处理 {i//batch_size + 1} 计算失败: {e}")
            continue
    
    if total_tokens == 0:
        print("所有批处理的困惑度计算失败")
        return {"mean_perplexity": float('nan'), "mean_loss": float('nan')}
    
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
    print(f"{model_name}总结果：")
    print(results)
    
    return results

from evaluate import load
from transformers import pipeline
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import math
from peft import PeftModel
MAX_LENGTH = 1024

dataset = load_dataset("json", data_files="/root/autodl-tmp/SFT/LoRA/LoRA_data.jsonl", split="train")
# 划分训练集和验证集
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
tokenizer = AutoTokenizer.from_pretrained("./Baichuan2-7B-Base/", use_fast=False , trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained("./Baichuan2-7B-Base/" , torch_dtype = torch.float16 , device_map = "auto" ,  trust_remote_code=True)
model = PeftModel.from_pretrained(base_model, "./lora_legal_qa_adapter")
baseline_results = calculate_perplexity_conservative(eval_dataset, batch_size = 2 , device= "cuda" if torch.cuda.is_available() else "cpu" , model = base_model , tokenizer = tokenizer , optimized = False)
optimized_results = calculate_perplexity_conservative(eval_dataset, batch_size = 2 , device= "cuda" if torch.cuda.is_available() else "cpu" , model = model , tokenizer = tokenizer , optimized = True)

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
for key, value in improvement.items():
    if isinstance(value, dict):
        print(f"{key}:")
        for k, v in value.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    else:
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
print("\n" + "=" * 30)
