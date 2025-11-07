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

from utils.set_logger import set_logger
from src.config import Config

# 设置logger
logger = set_logger('perplexity_evaluation.log')


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
                if len(input_id) > Config.tokenizer_max_length:
                    input_id = input_id[:Config.tokenizer_max_length]
                    attention_mask = attention_mask[:Config.tokenizer_max_length]
                    label = label[:Config.tokenizer_max_length]
                    logger.debug(f"文本被截断，长度: {len(input_id)}")
                else:
                    padding_length = Config.tokenizer_max_length - len(input_id)
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
    
    dataset = load_dataset("json", data_files=Config.lora_data_path, split="train")
    split_dataset = dataset.train_test_split(test_size=0.1, seed=Config.seed)
    train_dataset = split_dataset["train"]
    test_dataset = split_dataset["test"]
    logger.info(f"测试数据集大小: {len(test_dataset)}")
    
    tokenizer = AutoTokenizer.from_pretrained(Config.baichuan_model_path, use_fast=False, trust_remote_code=True)
    
    # 确保tokenizer有pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 加载基线模型
    logger.info("加载基线模型")
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.baichuan_model_path, 
        dtype=Config.dtype, 
        device_map=Config.device, 
        trust_remote_code=True
    )
    base_model.eval()
    
    # 加载LoRA模型
    logger.info("加载LoRA模型")
    lora_model = AutoModelForCausalLM.from_pretrained(
        Config.baichuan_model_path, 
        dtype=Config.dtype, 
        device_map=Config.device, 
        trust_remote_code=True
    )
    lora_model = PeftModel.from_pretrained(lora_model, Config.lora_adapter_path)
    lora_model.eval()
    
    device = Config.device
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