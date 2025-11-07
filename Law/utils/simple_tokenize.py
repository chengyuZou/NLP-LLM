import jieba
from typing import List
def simple_tokenize(text: str) -> List[str]:
    """
    简单中文分词

    Args:
        text (str): 输入中文文本


    Returns:
        List[str]: 分词结果
    """
    return list(jieba.cut_for_search(text))