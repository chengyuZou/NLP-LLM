import os
import json
import logging
import jieba
import re
import random
from typing import List, Tuple, Dict, Any, Optional

import torch
import streamlit as st  # 只是为了 @st.cache_resource
import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from rouge_chinese import Rouge

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from peft import PeftModel

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Config
from utils.set_logger import set_logger


# --- 日志配置 ---
logger = set_logger("evaluate_rag.log")


# ----------------- 1. 模型加载 (复用) -----------------
# (使用 @st.cache_resource 只是为了方便复用 app.py 的逻辑，
#  在脚本中它等同于一个普通的缓存)
@st.cache_resource
def load_all_components(config: Config) -> Dict[str, Any]:
    """
    一次性加载所有组件。包括llm_pipeline embedding_model rerank_model index_faiss index_bm25 chunks

    Args:
        config: Config 配置文件

    Returns:
        Dict[str, Any]: 组件字典
    """
    logger.info("--- 开始加载所有模型和索引 ---")
    components = {}
    try:
        # 1. 加载 LLM Pipeline
        logger.info("加载 LLM Pipeline...")
        dtype = config.dtype
        # 1.1 首先加载base_model 即Baichuan2-7B-base
        base_model = AutoModelForCausalLM.from_pretrained(
            config.baichuan_model_path, dtype=dtype, device_map=config.device, trust_remote_code=True
        )
        # 1.2 如果路径下存在LoRA的适配器,则加载,否则model不变
        if config.new_apapter_output_dir and os.path.exists(config.new_apapter_output_dir):
            model = PeftModel.from_pretrained(base_model, config.new_apapter_output_dir)
        else:
            model = base_model
        # 1.3 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.baichuan_model_path, trust_remote_code=True, use_fast=False)
        # 1.4 获得第一个组件, llm_pipeline
        components["llm_pipe"] = pipeline(
            config.task, model=model, tokenizer=tokenizer, device_map=config.device,
            dtype=dtype, max_new_tokens = config.max_new_tokens, do_sample=config.do_sample, temperature=config.temperature,
            top_p=config.top_p, repetition_penalty=config.repetition_penalty,
            pad_token_id=getattr(tokenizer, "eos_token_id", tokenizer.pad_token_id),
        )

        # 2. 加载 Embedding Model
        logger.info("加载 Embedding Model (BGE-Large)...")
        components["emb_model"] = HuggingFaceEmbeddings(
            model_name=config.bge_large_zh_model_path,
            model_kwargs={"device": config.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        # 3. 加载 Reranker Model
        logger.info("加载 Reranker Model (BGE-Rerank)...")
        components["rerank_tokenizer"] = AutoTokenizer.from_pretrained(config.bge_rerank_model_path)
        components["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(config.bge_rerank_model_path)
        components["rerank_model"].to(config.device).eval()

        # 4. 加载 FAISS 索引
        logger.info("加载 FAISS 索引...")
        components["vector_db"] = FAISS.load_local(
            config.faiss_index_dir, components["emb_model"], allow_dangerous_deserialization=True
        )

        # 5. 加载 BM25 索引
        logger.info("加载并初始化 BM25 索引...")
        with open(config.tokenized_corpus_path, "r", encoding="utf-8") as f:
            tokenized_texts = json.load(f)
        components["bm25"] = BM25Okapi(tokenized_texts)

        # 6. 加载 Chunks 语料库
        logger.info("加载 Chunks 语料库...")
        chunks = []
        with open(config.chunks_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
        components["chunks"] = chunks
        
        logger.info(f"--- 所有 {len(components)} 个组件加载完毕 ---")
        return components

    except Exception as e:
        logger.error(f"加载组件时发生致命错误: {e}", exc_info=True)
        raise

# ----------------- 2. 检索逻辑 (复用) -----------------
from utils.retrieve_bm25_topk import retrieve_bm25_topk
from utils.retrieve_emb_topk import retrieve_emb_topk
from utils.rrf_fusion import rrf_fusion
from utils.multi_retrieve_and_rerank import multi_retrieve_and_rerank

def compose_prompt_with_context(question: str, docs: List[Document]) -> str:
    """
    (共享) 简单 prompt 拼接策略。
    """
    ctxs = [f"[{i+1}] {d.page_content.strip()}" for i, d in enumerate(docs)]
    context_block = "\n\n".join(ctxs)
    return (
        "你是一个具有法律专业知识的智能助手。请仅基于下面提供的上下文(Context)回答用户的问题，"
        "并在答案末尾列出你引用的文档编号。\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    )

# ----------------- 3. 评测主逻辑 -----------------

def parse_qa_from_chunk(chunk: Document) -> Optional[Tuple[str, str]]:
    """
    从 "问题:...\n 答案:..." 格式的 chunk 文本中提取 Q 和 A。
    """
    match = re.search(r"问题:(.*?)\n 答案:(.*)", chunk.page_content, re.DOTALL)
    if match:
        question = match.group(1).strip()
        answer = match.group(2).strip()
        if question and answer:
            return question, answer
    logger.warning(f"无法从 chunk_index {chunk.metadata.get('chunk_index')} 中解析 Q/A")
    return None

def evaluate(config: Config):
    """
    执行 RAG 评测的主函数。
    """
    logger.info("--- 开始 RAG 评测流程 ---")
    
    # 1. 加载所有组件
    try:
        components = load_all_components(config)
    except Exception as e:
        logger.error(f"加载评测组件失败: {e}", exc_info=True)
        return

    # 2. 准备评测集
    # 我们从完整的 chunks 语料库中随机抽样
    all_chunks = components["chunks"]
    if len(all_chunks) < config.evaluation_sample_size:
        logger.warning(f"语料库大小 ({len(all_chunks)}) 小于请求的评测集大小 ({config.evaluation_sample_size})。将使用所有语料库。")
        test_set = all_chunks
    else:
        # 随机抽样
        test_set = random.sample(all_chunks, config.evaluation_sample_size)
    
    logger.info(f"评测集准备完毕，共 {len(test_set)} 条。")

    # 3. 初始化评测指标
    k = config.eval_k
    hit_rate_k = 0
    mrr_k = 0
    total_rouge_l_f = 0
    processed_count = 0
    rouge = Rouge()

    # 4. 遍历评测集
    for chunk in tqdm(test_set, desc="正在评测 RAG Pipeline"):
        # 4.1. 提取 Q/A 和 GT Chunk ID
        qa_pair = parse_qa_from_chunk(chunk)
        if not qa_pair:
            continue
        
        # 获取 QA Pair
        query, gt_answer = qa_pair
        # 获取当前chunk索引
        gt_chunk_index = chunk.metadata.get("chunk_index")
        if gt_chunk_index is None:
            logger.warning("Chunk 缺少 'chunk_index'，跳过。")
            continue
        
        # 4.2. 执行 RAG
        # (我们在这里评测 "高级 RAG" 模式)
        retrieved_docs = multi_retrieve_and_rerank(
            query=query,
            vector_db=components["vector_db"],
            bm25=components["bm25"],
            chunks=components["chunks"],
            rerank_tokenizer=components["rerank_tokenizer"],
            rerank_model=components["rerank_model"],
            config=config,
            k_final=k  # 最终召回 K 个
        )

        # 4.3. 计算检索器 (Retriever) 指标
        # 获取召回文档的index
        retrieved_indices = [doc.metadata.get("chunk_index") for doc in retrieved_docs]
        
        # 如果当前的index 在召回文档索引中
        if gt_chunk_index in retrieved_indices:
            # 命中率 +1
            hit_rate_k += 1
            # .index() 找到的是 0-based 排名
            rank = retrieved_indices.index(gt_chunk_index) + 1
            mrr_k += (1.0 / rank)
        
        # 4.4. 计算生成器 (Generator) 指标
        if not retrieved_docs:
            # 如果检索器什么都没返回，我们还是得生成一个"答案"
            generated_answer = "抱歉，未能检索到相关信息。"
        else:
            # 使用生成器生成答案
            prompt = compose_prompt_with_context(query, retrieved_docs)
            try:
                # 调用 LLM Pipeline,获得生成text
                out = components["llm_pipe"](prompt)
                raw_text = out[0].get("generated_text", "")
                # 如果text以prompt开头，则截取
                if raw_text.startswith(prompt):
                    generated_answer = raw_text[len(prompt):].strip()
                # 否则 不管
                else:
                    generated_answer = raw_text.strip()
            except Exception as e:
                logger.error(f"LLM 生成失败: {e}")
                generated_answer = ""

        # 4.5. 计算 ROUGE
        # 如果生成的答案和真实答案都不为空
        if generated_answer and gt_answer:
            try:
                # 计算 ROUGE
                scores = rouge.get_scores(generated_answer, gt_answer)
                total_rouge_l_f += scores[0]['rouge-l']['f']
            except Exception as e:
                logger.warning(f"ROUGE 计算失败: {e}")

        processed_count += 1
        
    if processed_count == 0:
        logger.error("评测集处理失败,0 条数据被成功处理。")
        return

    # 5. 汇总并打印结果
    avg_hit_rate_k = (hit_rate_k / processed_count) * 100
    avg_mrr_k = mrr_k / processed_count
    avg_rouge_l_f = (total_rouge_l_f / processed_count) * 100

    print("\n\n" + "="*30)
    print("      RAG 评测结果 (高级 Pipeline)     ")
    print("="*30)
    print(f" 评测集大小: {processed_count} 条")
    print(f" K 值 (Top-K): {k}")
    print("\n--- 检索器 (Retriever) 评测 ---")
    print(f" 命中率 (Hit Rate) @{k}: {avg_hit_rate_k:.2f} %")
    print(f" 平均倒数排名 (MRR) @{k}: {avg_mrr_k:.4f}")
    print("\n--- 生成器 (Generator) 评测 ---")
    print(f" ROUGE-L (F1 Score): {avg_rouge_l_f:.2f} %")
    print("="*30)

if __name__ == "__main__":
    # 设置随机种子以保证评测可复现
    random.seed(42)
    
    config = Config()
    evaluate(config)
