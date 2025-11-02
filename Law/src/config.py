import torch

class Config:
    data_path = "../data/DISC-Law-SFT-Pair-QA-released.jsonl"
    triplet_data_path = "../data/DISC-Law-Triplet-QA-released.jsonl"
    lora_data_path = "../data/lora_data.jsonl"

    baichuan_model_path = "../models/baichuan2-7b-base/"
    bge_large_zh_model_path = "../models/BAAI/bge-large-zh"
    bge_rerank_model_path = "../models/BAAI/bge-reranker-zh"
    dtype = torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer_max_length = 1024

    lora_r = 8 # lora rank
    lora_alpha = 32 # lora alpha
    lora_dropout = 0.1 # lora dropout
    lora_target_modules = ["o_proj", "gate_proj", "down_proj"] # 适配器模块

    apapter_output_dir = "../models/lora_legal_qa_adapter" # 保存适配器的目录路径
    new_apapter_output_dir = "../models/lora_new_legal_qa_adapter"
    overwrite_output_dir = True, # 覆盖输出目录中的内容
    num_train_epochs = 3 # 训练轮数
    per_device_train_batch_size = 4 # 每个设备的训练批次大小
    per_device_eval_batch_size = 10 # 每个设备的评估批次大小
    learning_rate = 1e-5 # 学习率
    gradient_accumulation_steps = 10 # 梯度累积步数（模拟更大的批次大小）
    logging_steps = 10 # 每多少步记录一次日志
    save_steps = 500 # 保存间隔步数
    save_strategy = "steps", # 按步数保存策略
    eval_steps = 500 # 评估间隔步数
    eval_strategy = "steps", # 按步数评估策略
    metric_for_best_model = "eval_loss" # 用于选择最佳模型的指标
    warmup_steps = 100 # 学习率预热步数
    max_grad_norm = 1.0 # 梯度裁剪的最大范数

    # pipeline
    task = "text-generation"
    max_new_tokens = 512
    do_sample=True
    temperature=0.7
    top_p=0.9
    repetition_penalty=1.1

    # 新增的产物路径
    faiss_index_dir = "../index/faiss_legal_qa_index"
    tokenized_corpus_path = "../index/tokenized_corpus.json"
    chunks_corpus_path = "../index/chunks_corpus.jsonl"
    chunk_size = 1024
    chunk_overlap = 128
  
    # 检索超参数
    topk_simple = 5      # v1: 标准 RAG 检索 5 条
    topk_bm25 = 50       # v2: BM25 召回
    topk_emb = 50        # v2: Embedding 召回
    topk_combined = 50   # v2: RRF 融合后
    topk_rerank = 5      # v2: 最终 Rerank 后取 5 条
    rrf_k = 60

    # 评测集大小
    rag_evaluation_sample_size = 100 # 从语料库中随机抽取200条进行评测

    # 设置随机数种子
    seed = 42 