import os
import json
import logging
import jieba
from typing import List, Tuple, Dict, Any, Optional

import torch

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Config

# --- 日志配置 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("build_all_indexes")

# --- 复用 v2 pipeline 的函数 ---

def load_jsonl_as_documents(path: str) -> List[Document]:
    docs = []
    logger.info(f"开始从 {path} 加载文档...")
    try:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    item = json.loads(line)
                    question = item.get("input", "")
                    answer = item.get("output", "")
                    meta = {"id": item.get("id", f"line_{i}"), "line": i, "source_path": path}
                    text = f"问题:{question}\n 答案:{answer}"
                    docs.append(Document(page_content=text, metadata=meta))
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {i} 行 JSON 解析失败，已跳过。错误: {e}")
    except FileNotFoundError:
        logger.error(f"错误: 数据文件未找到: {path}")
        raise
    logger.info(f"从 {path} 路径下加载了 {len(docs)} 条文档")
    return docs

def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    logger.info(f"开始分块... 原始文档数: {len(docs)}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "!", "？", "；", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    logger.info(f"分块完成。原始文档: {len(docs)} -> 分块: {len(chunks)}")
    return chunks

def init_embedding_model(model_path: str, device: str) -> HuggingFaceEmbeddings:
    logger.info(f"开始加载 Embedding 模型: {model_path}")
    try:
        emb_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        logger.info("Embedding 模型加载成功。")
        return emb_model
    except Exception as e:
        logger.error(f"错误: 无法加载 Embedding 模型 {model_path}。错误: {e}", exc_info=True)
        raise

def simple_tokenize(text: str) -> List[str]:
    return list(jieba.cut_for_search(text))

# --- 主构建函数 ---
def main():

    # 1. 加载和分块
    docs = load_jsonl_as_documents(Config.data_path)
    if not docs:
        logger.error("未能加载任何文档，程序终止。")
        return
        
    chunks = chunk_documents(docs, Config.chunk_size, Config.chunk_overlap)
    if not chunks:
        logger.error("未能对文档进行分块，程序终止。")
        return

    # 2. 加载嵌入模型
    emb_model = init_embedding_model(Config.embmodel_path, Config.device)

    # 3. 构建并保存 FAISS 
    if os.path.exists(Config.faiss_index_dir) and os.listdir(Config.faiss_index_dir):
        logger.info(f"从 {Config.faiss_index_dir} 加载已有的 FAISS 索引...")
        vector_db = FAISS.load_local(Config.faiss_index_dir, emb_model, allow_dangerous_deserialization=True)
    else:
        logger.info(f"构建 FAISS 索引... (这可能需要一些时间)")
        vector_db = FAISS.from_documents(chunks, emb_model)
        vector_db.save_local(Config.faiss_index_dir)
        logger.info(f"FAISS 索引已保存到 {Config.faiss_index_dir}")

    # 4. 构建并保存 BM25 所需的分词语料
    logger.info("为 BM25 构建分词语料...")
    texts = [c.page_content for c in chunks]
    tokenized_texts = [simple_tokenize(text) for text in texts]
    try:
        with open(Config.tokenized_corpus_path, "w", encoding="utf-8") as f:
            json.dump(tokenized_texts, f)
        logger.info(f"BM25 分词语料已保存到 {Config.tokenized_corpus_path}")
    except Exception as e:
        logger.error(f"保存 BM25 语料失败: {e}", exc_info=True)

    # 5. 保存 Reranker 所需的 Chunks 语料
    logger.info("保存 Chunks 语料 (用于 Reranker)...")
    try:
        with open(Config.chunks_corpus_path, "w", encoding="utf-8") as f:
            for c in chunks:
                # 序列化 Document 对象
                serializable_chunk = {
                    "page_content": c.page_content,
                    "metadata": c.metadata
                }
                f.write(json.dumps(serializable_chunk, ensure_ascii=False) + "\n")
        logger.info(f"Chunks 语料已保存到 {Config.chunks_corpus_path}")
    except Exception as e:
        logger.error(f"保存 Chunks 语料失败: {e}", exc_info=True)

    logger.info("--- 所有索引和语料构建完毕 ---")

if __name__ == "__main__":
    main()
