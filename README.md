# NLP-LLM
##THUNews-classification：中文文本分类＋可视化演示工程项目

1	- 下载 THUCNews ，获取子集（5 类×2000 条）
- 用 pandas / jsonlines 做数据清洗	数据清洗（缺失 / 标签平衡）

2	- 用 SentencePiece / Byte-Pair Encoding 训练中英文混合子词
- 对比 vocab-size 对性能的影响(8k , 16k, 32k)
- 子词模型原理 & 超参调优
  
3	- 封装 datasets.Dataset ，实现自定义 DataCollator 
- 可视化样本长度分布、Token 分布	datasets 深度用法

4	- 用 Hugging Face 的 Trainer API 微调 bert-base-chinese
- 输出 baseline Validation Accuracy / F1	Trainer 全流程
