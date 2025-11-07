import os
import json
import logging
from typing import List, Tuple, Dict, Any, Optional
import jieba

import torch
import streamlit as st
import numpy as np
from rank_bm25 import BM25Okapi

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from peft import PeftModel

from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.config import Config
from utils.set_logger import set_logger

# --- 日志配置 ---
logger = set_logger("legal_rag_app_v2")

# ----------------- 1. 所有模型的加载函数 (缓存) -----------------

@st.cache_resource
def load_all_components(config: Config) -> Dict[str, Any]:
    """
    使用 st.cache_resource 一次性加载所有模型和索引。包括llm_pipeline embedding_model rerank_model index_faiss index_bm25 chunks

    Args:
        config: Config 配置文件

    Returns:
        Dict[str, Any]: 组件字典
    """
    
    logger.info("--- 开始加载所有模型和索引 (缓存) ---")
    components = {}
    
    try:
        # 1. 加载 LLM Pipeline (来自 v1)
        logger.info("加载 LLM Pipeline...")
        dtype = torch.float16 if config.device.startswith("cuda") else torch.float32
        base_model = AutoModelForCausalLM.from_pretrained(
            config.baichuan_model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
        )
        if config.new_apapter_output_dir and os.path.exists(config.new_apapter_output_dir):
            model = PeftModel.from_pretrained(base_model, config.new_apapter_output_dir)
            logger.info(f"LoRA adapter 从 {config.new_apapter_output_dir} 加载成功")
        else:
            model = base_model
        tokenizer = AutoTokenizer.from_pretrained(config.baichuan_model_path, trust_remote_code=True, use_fast=False)
        components["llm_pipe"] = pipeline(
            "text-generation", model=model, tokenizer=tokenizer, device_map="auto",
            torch_dtype=dtype, max_new_tokens=512, do_sample=True, temperature=0.7,
            top_p=0.9, repetition_penalty=1.1,
            pad_token_id=getattr(tokenizer, "eos_token_id", tokenizer.pad_token_id),
        )

        # 2. 加载 Embedding Model (来自 v2)
        logger.info("加载 Embedding Model (BGE-Large)...")
        components["emb_model"] = HuggingFaceEmbeddings(
            model_name=config.bge_large_zh_model_path,
            model_kwargs={"device": config.device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

        # 3. 加载 Reranker Model (来自 v2)
        logger.info("加载 Reranker Model (BGE-Rerank)...")
        components["rerank_tokenizer"] = AutoTokenizer.from_pretrained(config.bge_rerank_model_path)
        components["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(config.bge_rerank_model_path)
        components["rerank_model"].to(config.device).eval()

        # 4. 加载 FAISS 索引
        logger.info("加载 FAISS 索引...")
        components["vector_db"] = FAISS.load_local(
            config.faiss_index_dir, components["emb_model"], allow_dangerous_deserialization=True
        )

        # 5. 加载 BM25 索引 (通过 tokenized 语料库)
        logger.info("加载并初始化 BM25 索引...")
        with open(config.tokenized_corpus_path, "r", encoding="utf-8") as f:
            tokenized_texts = json.load(f)
        components["bm25"] = BM25Okapi(tokenized_texts)

        # 6. 加载 Chunks 语料库
        logger.info("加载 Chunks 语料库...")
        chunks = []
        with open(config.chunks_corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunks.append(Document(page_content=data["page_content"], metadata=data["metadata"]))
        components["chunks"] = chunks
        
        logger.info(f"--- 所有 {len(components)} 个组件加载完毕 ---")
        return components

    except Exception as e:
        logger.error(f"加载组件时发生致命错误: {e}", exc_info=True)
        st.error(f"模型/索引加载失败: {e}\n请检查路径: {config.base_model_path}, {config.bge_large_zh_model_path}, {config.bge_rerank_model_path}, {config.faiss_index_dir}, {config.tokenized_corpus_path}, {config.chunks_corpus_path}")
        st.stop()


# ----------------- 2. RAG 逻辑函数 -----------------

# --- 2a. 共享的 Prompt 模板 ---
def compose_prompt_with_context(question: str, docs: List[Document]) -> str:
    """
    (共享) 简单 prompt 拼接策略。
    """
    ctxs = []
    for i, d in enumerate(docs, 1):
        snippet = d.page_content.strip()
        ctxs.append(f"[{i}] {snippet}")
    context_block = "\n\n".join(ctxs)
    
    prompt = (
        "你是一个具有法律专业知识的智能助手。请仅基于下面提供的上下文(Context)回答用户的问题，"
        "并在答案末尾列出你引用的文档编号。\n\n"
        f"Context:\n{context_block}\n\nQuestion: {question}\n\nAnswer:"
    )
    return prompt

# --- 2b. 标准 RAG 逻辑 (v1) ---
def answer_by_rag_simple(
    llm_pipe, 
    vector_db: FAISS, 
    question: str, 
    k: int
) -> Tuple[str, List[Document]]:
    """
    执行标准 RAG:
    1. FAISS 检索
    2. 构建 Prompt
    3. LLM 生成
    """
    logger.info("执行 [标准 RAG] 流程...")
    # 1. 检索
    retrieved = vector_db.similarity_search(question, k=k)
    
    # 2. 构建 prompt
    prompt = compose_prompt_with_context(question, retrieved)

    # 3. 生成
    out = llm_pipe(prompt)
    raw_text = out[0].get("generated_text", "")
    if raw_text.startswith(prompt):
        answer = raw_text[len(prompt):].strip()
    else:
        answer = raw_text.strip()

    return answer, retrieved

# --- 2c. 高级 RAG 逻辑 (v2) ---
from utils.retrieve_bm25_topk import retrieve_bm25_topk
from utils.retrieve_emb_topk import retrieve_emb_topk
from utils.rrf_fusion import rrf_fusion
from utils.multi_retrieve_and_rerank import multi_retrieve_and_rerank

def answer_by_rag_advanced(
    llm_pipe, 
    vector_db: FAISS, 
    bm25: BM25Okapi,
    chunks: List[Document],
    rerank_tokenizer: AutoTokenizer,
    rerank_model: AutoModelForSequenceClassification,
    config: Config,
    question: str
) -> Tuple[str, List[Document]]:
    """
    执行高级 RAG:
    1. v2 检索 (RRF + Rerank)
    2. 构建 Prompt
    3. LLM 生成
    """
    # 1. v2 检索
    retrieved_docs = multi_retrieve_and_rerank(
        question, vector_db, bm25, chunks, rerank_tokenizer, rerank_model, config
    )
    
    # 2. 构建 prompt
    prompt = compose_prompt_with_context(question, retrieved_docs)

    # 3. 生成
    out = llm_pipe(prompt)
    raw_text = out[0].get("generated_text", "")
    if raw_text.startswith(prompt):
        answer = raw_text[len(prompt):].strip()
    else:
        answer = raw_text.strip()

    return answer, retrieved_docs

# ----------------- 3. Streamlit UI 界面 -----------------

# --- 页面配置 ---
st.set_page_config(
    page_title="法律问答大模型",
    page_icon="⚖️",
    initial_sidebar_state="auto"
)

# --- CSS 样式 (来自你的 minimind 示例) ---
st.markdown("""
    <style>
        /* (这里省略了你提供的长串 CSS，保持原样) */
        /* ... 你提供的所有 .stButton, .stMainBlockContainer, .stApp 样式 ... */
        
        /* 来源文档的样式 */
        .source-container {
            border-top: 1px solid #eee;
            margin-top: 15px;
            padding-top: 10px;
        }
        .source-item {
            font-size: 0.9em;
            color: #555;
            background-color: #f9f9f9;
            border-radius: 5px;
            padding: 8px;
            margin-bottom: 5px;
            border: 1px solid #eee;
        }
        .source-item summary {
            font-weight: bold;
            cursor: pointer;
        }
    </style>
""", unsafe_allow_html=True)

# --- 标题 ---
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    f'<span style="font-size: 26px; font-weight: 900; margin-left: 10px;">⚖️ 法律 RAG 问答 (双模式)</span>'
    '</div>',
    unsafe_allow_html=True
)

# --- 侧边栏 ---
st.sidebar.title("🛠️ RAG 模式设置")
st.sidebar.toggle(
    "🔬 深度检索 (高级RAG)", 
    value=False, 
    key="deep_rag_toggle",
    help="开启后，将使用 BM25+Embedding+RRF+Rerank 的高级检索模式，速度较慢但可能更准。关闭则使用快速的 FAISS 检索。"
)
st.sidebar.markdown("---")
if st.sidebar.button("清空聊天记录", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# --- 加载所有组件 ---
config = Config()
try:
    components = load_all_components(config)
    # 在侧边栏显示加载成功状态
    st.sidebar.success("所有模型和索引加载成功！")
    st.sidebar.markdown(f"**LLM**: {config.base_model_path}\n"
                        f"**Reranker**: {config.rerank_model_path}\n"
                        f"**Chunks**: {len(components['chunks'])} 条", 
                        unsafe_allow_html=True)
except Exception:
    # 错误已在加载函数中通过 st.error 和 st.stop 处理
    pass


# --- 聊天界面 ---

if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    avatar = "⚖️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"], unsafe_allow_html=True)

# 获取用户输入
if prompt := st.chat_input("请输入你的法律问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 根据 Toggle 状态决定调用哪个 RAG 流程
    is_deep_rag = st.session_state.deep_rag_toggle
    
    with st.chat_message("assistant", avatar="⚖️"):
        placeholder = st.empty()
        
        try:
            if is_deep_rag:
                # --- 高级 RAG 流程 (v2) ---
                placeholder.markdown("🔬 **深度检索中...** (执行 高级RAG)")
                answer, sources = answer_by_rag_advanced(
                    llm_pipe=components["llm_pipe"],
                    vector_db=components["vector_db"],
                    bm25=components["bm25"],
                    chunks=components["chunks"],
                    rerank_tokenizer=components["rerank_tokenizer"],
                    rerank_model=components["rerank_model"],
                    config=config,
                    question=prompt
                )
            else:
                # --- 标准 RAG 流程 (v1) ---
                placeholder.markdown("🔍 **标准检索中...** (执行 初级RAG)")
                answer, sources = answer_by_rag_simple(
                    llm_pipe=components["llm_pipe"],
                    vector_db=components["vector_db"],
                    question=prompt,
                    k=config.topk_simple
                )

            # 格式化并显示助手答案 + 来源
            full_response = f"{answer}\n\n"
            if sources:
                full_response += '<div class="source-container"><strong>参考来源：</strong>\n'
                for i, doc in enumerate(sources, 1):
                    source_id = doc.metadata.get('id', f'doc_{i}')
                    snippet = doc.page_content.replace('\n', ' ').strip()
                    snippet_preview = snippet[:150] + "..." if len(snippet) > 150 else snippet
                    
                    full_response += (
                        f'<details class="source-item">'
                        f'<summary>来源 [{i}] (ID: {source_id})</summary>'
                        f'<div>{snippet_preview}</div>'
                        f'</details>\n'
                    )
                full_response += '</div>'
            else:
                full_response += '<div class="source-container"><strong>未能检索到相关上下文。</strong></div>'

            # 更新占位符
            placeholder.markdown(full_response, unsafe_allow_html=True)
            # 存入历史记录
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            logger.error(f"在处理查询 '{prompt}' 时出错: {e}", exc_info=True)
            placeholder.error(f"处理您的请求时出现错误: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"错误: {e}"})

# streamlit run app_with_toggle.py --server.address=127.0.0.1 --server.port=6006
