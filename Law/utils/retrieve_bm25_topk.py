from rank_bm25 import BM25Okapi
from typing import List, Tuple
import numpy as np

from utils.simple_tokenize import simple_tokenize
def retrieve_bm25_topk(bm25: BM25Okapi, query: str, top_k: int) -> List[Tuple[int, float]]:
    """
    从文档中由BM25算法检索出前k个最相关的文档
    
    Args:
        bm25: BM25模型
        query: 查询文本
        top_k: 返回最相关的文档数量


    Returns:
        List[Tuple[int, float]]: 返回最相关的文档的索引和得分
    """
    try:
        # 先对query分词
        q_tokens = simple_tokenize(query)
        # 获取BM25算法的得分
        scores = BM25Okapi(q_tokens)
        # 获取前k个索引
        # 这里先进行np.argsort(scores)获取顺序排列的索引 eg: np.argsort([1,2,3,0,5]) = [4,0,1,2,3]
        # 随后,[::-1]表示取倒序 eg: [4,0,1,2,3][::-1] = [3,2,1,0,4]
        # 最后获取前k个索引 eg: [3,2,1,0,4][:3] = [3,2,1]
        index = np.argsort(scores)[::-1][top_k]
        # 获得前k个得分
        scores = [scores[i] for i in index]
        # 获取输出 输出为索引和得分
        return list(zip(index.astype(int) , scores.astype(float)))
    
    except Exception as e: 
        
        return []