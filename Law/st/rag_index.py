import os
import json
import logging
from typing import List, Tuple, Optional

import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_index")

# -------------------- config (请核对这些路径) --------------------
# 原始 SFT jsonl（id,input,output,...）
DATA_JSONL = "/root/autodl-tmp/SFT/Triplet/DISC-Law-SFT-Triplet-QA-released.jsonl"
# embeddings 模型路径
EMB_MODEL_PATH = "../bge-large-zh-v1.5/"
# 索引保存路径 (app.py 会从这里加载)
FAISS_INDEX_DIR = "faiss_legal_qa_index"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------------------------

def load_documents_from_jsonl(path: str) -> List[Document]:
    """
    从 JSONL 文件加载数据为 LangChain Document 列表。
    与你的 rag_manual_v1.py 中的函数相同。
    """
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                j = json.loads(line)
            except Exception as e:
                logger.warning("跳过第 %d 行，JSON 解析失败: %s", i, e)
                continue
            q = j.get("input", "").strip() + "\n"
            a = j.get("output", "").strip()
            text = f"问题：{q}\n答案：{a}"
            meta = {"id": j.get("id", f"line_{i}"), "line": i, "source": "legal_qa"}
            docs.append(Document(page_content=text, metadata=meta))
    logger.info("加载 JSONL 完成，文档数=%d", len(docs))
    return docs

def split_documents(docs: List[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    """
    将文档分割成块。
    与你的 rag_manual_v1.py 中的函数相同。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)
    logger.info("分块完成：%d -> %d", len(docs), len(chunks))
    return chunks

def build_and_save_faiss(chunks: List[Document], emb_model_path: str, index_dir: str) -> None:
    """
    构建或加载 FAISS 索引。
    此版本*强制*构建并保存，覆盖旧的。
    """
    # 创建 embedding wrapper
    logger.info("加载嵌入模型: %s", emb_model_path)
    emb = HuggingFaceEmbeddings(
        model_name=emb_model_path,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

    if os.path.exists(index_dir):
        logger.warning("警告：检测到已有的索引目录 %s，将覆盖它。", index_dir)

    logger.info("开始构建 FAISS 索引（嵌入并存入）... 这可能需要一些时间。")
    vector_db = FAISS.from_documents(chunks, emb)
    vector_db.save_local(index_dir)
    logger.info("FAISS 索引构建完成并保存到 %s", index_dir)


def main():
    if not os.path.exists(DATA_JSONL):
        logger.error("错误：找不到数据文件 %s", DATA_JSONL)
        logger.error("请检查 DATA_JSONL 变量路径是否正确。")
        return
    if not os.path.exists(EMB_MODEL_PATH):
        logger.error("错误：找不到嵌入模型路径 %s", EMB_MODEL_PATH)
        logger.error("请检查 EMB_MODEL_PATH 变量路径是否正确。")
        return

    # 1. 加载原始数据
    docs = load_documents_from_jsonl(DATA_JSONL)
    if not docs:
        logger.error("未能从 %s 加载任何文档。", DATA_JSONL)
        return
        
    # 2. 分块
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    if not chunks:
        logger.error("未能将文档分块。")
        return

    # 3. build / save FAISS
    build_and_save_faiss(chunks, EMB_MODEL_PATH, FAISS_INDEX_DIR)


if __name__ == "__main__":
    main()
