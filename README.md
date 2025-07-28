# NLP-LLM
##THUNews-classification：中文文本分类＋可视化演示工程项目

1	- 下载 THUCNews ，获取子集（5 类×2000 条）
- 用 pandas / jsonlines 做数据清洗	数据清洗（缺失 / 标签平衡）

2	- 用 SentencePiece / Byte-Pair Encoding 训练中英文混合子词
- 对比 vocab-size 对性能的影响(8k , 16k, 32k)
- 子词模型原理 & 超参调优
