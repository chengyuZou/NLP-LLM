from typing import List

import torch
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.retrieve_bm25_topk import retrieve_bm25_topk
from utils.retrieve_emb_topk import retrieve_emb_topk
from utils.rrf_fusion import rrf_fusion
from src.config import Config

def multi_retrieve_and_rerank(
    query: str,
    vector_db: FAISS,
    bm25: BM25Okapi,
    chunks: List[Document],
    rerank_tokenizer: AutoTokenizer,
    rerank_model: AutoModelForSequenceClassification,
    config: Config,
    k_final: int # 最终需要 K 个
) -> List[Document]:
    """
    进行RRF + Rerank 检索，返回最终排序的 Document 列表.

    Args:
        query: 检索的查询文本
        vector_db: FAISS 向量数据库
        bm25: BM25 模型
        chunks: 文档列表
        rerank_tokenizer: Rerank 模型对应的 tokenizer
        rerank_model: Rerank 模型
        config: 配置参数
        k_final: 最终选得分最高的前K个文档

    Returns:
        List[Document]: 最终排序的 Document 文档
    """
    # 1. 粗召回

    # 1.1 BM25 检索
    bm25_list = retrieve_bm25_topk(bm25, query, config.topk_bm25)
    # 1.2 Embedding 检索
    emb_list = retrieve_emb_topk(vector_db, query, config.topk_emb)

    # 2. RRF 融合

    # 2.1获得rrf得分
    rrf_scores = rrf_fusion(bm25_list, emb_list, config.rrf_k)
    # 2.2 按照score进行倒叙排列并选取前k个
    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:config.topk_combined]
    
    # 3. 精排

    # 3.1 获取rrf排序后的文档id
    candidate_ids = [idx for idx, _ in sorted_rrf if idx >= 0 and idx < len(chunks)]
    if not candidate_ids: return []
    
    # 3.2 获取候选文档
    candidate_docs = [chunks[i] for i in candidate_ids]
    # 3.3 构建输入,组成rerank模型需要的pairs(query, doc)
    pairs = [(query, doc.page_content) for doc in candidate_docs]

    # 3.4 获取rerank模型需要的输入
    inputs = rerank_tokenizer(
        pairs, padding=True, truncation=True, return_tensors='pt', max_length=512
    ).to(config.device)

    # 3.5 获取rerank模型输出
    with torch.no_grad():
        scores = rerank_model(**inputs).logits.squeeze()
    
    # 3.6 构建rerank模型输出的映射关系
    scores_list = [scores.item()] if scores.dim() == 0 else scores.cpu().tolist()
    # 每一个doc对应一个score
    scored_candidates = list(zip(candidate_docs, scores_list))

    # 4. 最终排序
    sorted_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    
    # 5. 返回 topk_final 的 Document 对象
    return [doc for doc, score in sorted_candidates[:k_final]]