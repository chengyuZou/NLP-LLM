Legal-LLM-RAG/

├── app/

│   ├── app.py             # 你的 Streamlit UI 界面代码

│   └── (app_utils.py)     # (可选) 如果 UI 逻辑变复杂，可以把辅助函数放这里

│

├── data/

│   ├── .gitignore         # 自动忽略所有 .jsonl, .csv 等数据文件

│   └── README.md          # 说明需要下载哪些数据，并放到此目录下

│

├── index/

│   ├── .gitignore         # 自动忽略所有 faiss, json 等索引文件

│   └── README.md          # 说明此目录用于存放生成的 FAISS 和 BM25 索引

│

├── models/

│   ├── .gitignore         # 自动忽略所有模型文件

│   └── README.md          # 说明需要下载 Baichuan, BGE 等模型，并放到此目录

│

├── src/

│   │   

│   ├── __init__.py

│   ├── config.py          # 存放所有路径配置 (DATA_JSONL, BASE_MODEL_PATH 等)

│   └── prompt_builder.py  # 专用于构建 prompt 的函数 (如 compose_prompt_with_context)

│

├── scripts/

│   │   # 存放一次性的、用于驱动流程的脚本

│   ├── prepare_data.py  # 准备 LoRA_data.jsonl

│   ├── train_lora_sft.py  # 训练 LoRA 模型

│   ├── build_index.py   # 构建所有 RAG 索引

│   ├── evaluate_ppl.py  # 评测困惑度

│   ├── train_new_lora.py # 第二次训练LoRA 模型

│   └── evaluate_rag.py # 你的 RAGAS 

|

├── utils/

|   ├── __init__.py

|   ├── multi_retrieve_and_rerank.py # 多轮检索和rerank

|   ├── retrieve_bm25_topk.py # BM25检索

|   ├── retrieve_emb_topk.py # embedding检索

|   ├── rrf_fusion.py # RRF融合

|   ├── simple_tokenize.py  # 专用于分词的函数 (如 simple_tokenize)

│   └── set_logger.py # 设置日志 

│

├── requirements.txt       # 列出所有 Python 依赖包 (pip install -r requirements.txt)

└── README.md              # 你的新门面！(见下一个文件)

