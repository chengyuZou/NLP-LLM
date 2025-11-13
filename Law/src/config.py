import torch

class Config:
    # ====== 数据路径 ======
    data_path = "../data/DISC-Law-SFT-Pair-QA-released.jsonl"  # 训练数据路径
    triplet_data_path = "../data/DISC-Law-Triplet-QA-released.jsonl" # 三元组数据路径
    lora_data_path = "../data/lora_data.jsonl" # lora数据路径

    # ===== 模型路径与配置 ======
    baichuan_model_path = "../models/baichuan2-7b-base/" # Base 模型路径
    bge_large_zh_model_path = "../models/BAAI/bge-large-zh" # BGE Large ZH 模型路径,用与向量编码
    bge_rerank_model_path = "../models/BAAI/bge-reranker-zh" # BGE Reranker 模型路径
    dtype = torch.float16 # 数据类型
    device = "cuda" if torch.cuda.is_available() else "cpu" # 设备
    tokenizer_max_length = 1024 # tokenizer 最大长度

    # ===== LoRA 配置 ======
    lora_r = 8 # lora r
    lora_alpha = 32 # lora alpha
    lora_dropout = 0.1 # lora dropout
    lora_target_modules = ["o_proj", "gate_proj", "down_proj"] # 适配器模块

    # ===== 训练参数 ======
    adapter_output_dir = "../models/lora_legal_qa_adapter" # 保存适配器的目录路径
    new_adapter_output_dir = "../models/lora_new_legal_qa_adapter" # 使用三元组数据训练后的适配器路径
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

    # ===== Pipeline ======
    task = "text-generation" # 任务名称
    max_new_tokens = 512 # 最大生成长度
    do_sample=True # 是否使用采样
    temperature=0.7 # 采样温度
    top_p=0.9 # 采样 nucleus
    repetition_penalty=1.1 # 重复惩罚

    # ===== FAISS  ======
    faiss_index_dir = "../index/faiss_legal_qa_index" # FAISS 索引目录
    tokenized_corpus_path = "../index/tokenized_corpus.json" # tokenized_corpus.json
    chunks_corpus_path = "../index/chunks_corpus.jsonl" # chunks_corpus.jsonl
    chunk_size = 1024 # 分块大小
    chunk_overlap = 128 # 分块重叠长度
  
    # ===== 检索超参数 ======
    topk_simple = 5      # v1: 标准 RAG 检索 5 条
    topk_bm25 = 50       # v2: BM25 召回
    topk_emb = 50        # v2: Embedding 召回
    topk_combined = 50   # v2: RRF 融合后
    topk_rerank = 5      # v2: 最终 Rerank 后取 5 条
    rrf_k = 60           # RRF 融合参数
    eval_k = 5 

    # 评测集大小
    evaluation_sample_size = 200 # 从语料库中随机抽取200条进行评测

    # 设置随机数种子

    seed = 42 
