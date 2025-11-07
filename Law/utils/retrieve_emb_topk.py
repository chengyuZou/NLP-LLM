from langchain_community.vectorstores import FAISS
from typing import List, Tuple
import torch
def retrieve_emb_topk(vector_db: FAISS, query: str, top_k: int) -> List[Tuple[int, float]]:
    """
    从文档中由embedding模型检索出前k个最相关的文档
    
    Args:
        vector_db: FAISS向量数据库
        query: 查询文本
        top_k: 返回最相关的文档数量


    Returns:
        List[Tuple[int, float]]: 返回最相关的文档的索引和得分
        由于最后用RRF做召回只看rank 故有没有score也无所谓
    """
    try:
        # 从FAISS向量数据库中检索最相关的top_k个文档
        docs = vector_db.similarity_search(query,k = top_k)
        return [(int(doc.metadata["chunk_index"]), 0.0) for doc in docs if "chunk_index" in doc.metadata]
    except Exception: return []