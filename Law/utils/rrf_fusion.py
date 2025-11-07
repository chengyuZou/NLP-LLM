from typing import List , Dict , Tuple

def _list_to_rank_map(doc_list: List[Tuple[int, float]]) -> Dict[int, int]:
    """
    一个映射,输入为已经排序的 List[文档索引 , 文档得分]

    返回一个映射 文档索引 -> 文档rank
    """
    return {int(idx): rank for rank, (idx, _) in enumerate(doc_list)}
def rrf_fusion(bm25_list: List[Tuple[int, float]], emb_list: List[Tuple[int, float]], k_rrf: int) -> Dict[int, float]:
    """
    RRF多路召回算法,仅由查询到的文本的Rank决定
    
    Args:
        bm25_list: BM25算法返回的文档索引和得分
        emb_list: 嵌入模型返回的文档索引和得分
        k_rrf: rrf 融合参数


    Returns:
        Dict[int, float]: 融合后的文档索引和得分
    """
    try:
        rrf_scores = {}
        # 先获得映射后的文档索引
        bm25_rank = _list_to_rank_map(bm25_list)
        emb_rank = _list_to_rank_map(emb_list)
        # 计算RRF得分
        all_indices = set(bm25_rank.keys()) | set(emb_rank.keys())
        for indice in all_indices:
            score = 0.0
            # 如果索引在BM25中，则得分加1/（k_rrf + BM25得分）
            if indice in bm25_rank:
                score += 1.0 / (k_rrf + bm25_rank[indice])
            
            # 如果索引在Emb中，则得分加1/（k_rrf + Emb得分）
            if indice in emb_rank:
                score += 1.0 / (k_rrf + emb_rank[indice])
            
            # 映射
            rrf_scores[indice] = score
        # 返回
        return rrf_scores
    
    except Exception as e:
        return {}

