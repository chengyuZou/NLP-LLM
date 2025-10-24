# rag_manual_v1.py
import os
import json
import logging
from typing import List, Tuple, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# LangChain subpackages (these should work in your env)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_manual_v1")


# -------------------- config (请按需改) --------------------
DATA_JSONL = "/root/autodl-tmp/SFT/Triplet/DISC-Law-SFT-Triplet-QA-released.jsonl"   # 原始 SFT jsonl（id,input,output,...）
EMB_MODEL_PATH = "./bge-large-zh-v1.5/"                   # embeddings 模型
BASE_MODEL_PATH = "./Baichuan2-7B-Base/"                  # base 模型目录
LORA_ADAPTER_PATH = "./lora_new_sft_adapter/"             # LoRA adapter 目录（若无可设为 None）
FAISS_INDEX_DIR = "faiss_legal_qa_index"                  # 索引保存路径
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
# ----------------------------------------------------------


def load_documents_from_jsonl(path: str) -> List[Document]:
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


def build_or_load_faiss(chunks: List[Document], emb_model_path: str, index_dir: str) -> Tuple[FAISS, HuggingFaceEmbeddings]:
    # 创建 embedding wrapper
    emb = HuggingFaceEmbeddings(
        model_name=emb_model_path,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )
    # 若已有索引目录则 load，否则 build 并保存
    if os.path.exists(index_dir) and os.listdir(index_dir):
        logger.info("检测到已有 FAISS 索引，尝试加载：%s", index_dir)
        vector_db = FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
        logger.info("FAISS 索引加载完成")
    else:
        logger.info("构建 FAISS 索引（嵌入并存入）...")
        vector_db = FAISS.from_documents(chunks, emb)
        vector_db.save_local(index_dir)
        logger.info("FAISS 索引构建并保存到 %s", index_dir)
    return vector_db, emb


def init_generation_model(base_model_path: str, lora_adapter_path: Optional[str] = None):
    # 加载基础模型（仅用于推理 pipeline 包装）
    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    logger.info("加载 base 模型（dtype=%s，device=%s）", dtype, DEVICE)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    # 如有 LoRA adapter，加载
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        try:
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            logger.info("LoRA adapter 从 %s 加载成功", lora_adapter_path)
        except Exception as e:
            logger.warning("加载 LoRA 失败，使用 base model：%s", e)
            model = base_model
    else:
        model = base_model
    model.use_cache = False
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    # wrap pipeline (注意 device_map="auto" 可能把 model 分散到多卡)
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=getattr(tokenizer, "eos_token_id", tokenizer.pad_token_id),
    )
    logger.info("文本生成 pipeline 创建完成")
    return model, tokenizer, gen_pipe


def compose_prompt_with_context(question: str, docs: List[Document]) -> str:
    """
    简单 prompt 拼接策略 — 2-step RAG。
    你可以按需求替换成更复杂的模板（例如包含 <analysis> / <advice> 标签）
    """
    ctxs = []
    for i, d in enumerate(docs, 1):
        # 每个 doc 只保留前 N 字以防过长
        snippet = d.page_content.strip()
        ctxs.append(f"[{i}] {snippet}")
    context_block = "\n\n".join(ctxs)
    prompt = (
        "你是一个具有法律专业知识的智能助手。请仅基于下面提供的上下文（Context）回答用户的问题，"
        "并在答案末尾列出你引用的文档编号。\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    )
    return prompt


def answer_by_rag(gen_pipe, vector_db: FAISS, question: str, k: int = TOP_K) -> dict:
    # 检索
    retrieved = vector_db.similarity_search(question, k=k)
    logger.info("检索到 %d 条文档", len(retrieved))

    # 构建 prompt
    prompt = compose_prompt_with_context(question, retrieved)

    # 生成（pipeline 返回 list）
    out = gen_pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
    raw_text = out[0].get("generated_text", "")
    # 有些 pipeline 会返回 prompt+生成，去掉 prompt 部分
    if raw_text.startswith(prompt):
        answer = raw_text[len(prompt):].strip()
    else:
        answer = raw_text.strip()

    # 返回 answer + sources
    result = {
        "answer": answer,
        "raw": raw_text,
        "source_documents": [{"id": d.metadata.get("id"), "snippet": d.page_content[:800], "metadata": d.metadata} for d in retrieved]
    }
    return result


def main():
    # 1. 加载原始数据并分块（只在你需要重建索引时使用）
    docs = load_documents_from_jsonl(DATA_JSONL)
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # 2. build / load FAISS
    vector_db, emb_model = build_or_load_faiss(chunks, EMB_MODEL_PATH, FAISS_INDEX_DIR)

    # 3. 初始化模型 pipeline（LoRA 推理）
    model, tokenizer, gen_pipe = init_generation_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH)

    # 4. 示例查询
    q = "劳动合同法保护哪些权益？"
    res = answer_by_rag(gen_pipe, vector_db, q, k=5)

    print("==== ANSWER ====\n", res["answer"])
    print("\n==== SOURCES ====")
    for i, s in enumerate(res["source_documents"], 1):
        print(f"{i}. id={s['id']} meta={s['metadata']}")
        print(s["snippet"][:400].replace("\n", " "))
        print("----")


if __name__ == "__main__":
    main()
