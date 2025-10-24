import os
import re
import json
import logging
from typing import List, Tuple, Optional

import torch
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

# LangChain subpackages
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 日志配置 ---
# (在 Streamlit 中，日志主要输出到终端)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("legal_rag_app")

# -------------------- config (请核对这些路径) --------------------
# 嵌入模型 (必须与 build_index.py 中的一致)
EMB_MODEL_PATH = "../bge-large-zh-v1.5/"
# Base LLM 模型
BASE_MODEL_PATH = "../Baichuan2-7B-Base/"
# LoRA adapter 路径 (若无请设为 None)
LORA_ADAPTER_PATH = "../lora_new_sft_adapter/"
# 索引加载路径 (必须是 build_index.py 生成的)
FAISS_INDEX_DIR = "faiss_legal_qa_index"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5
# -----------------------------------------------------------------


# === 从 rag_manual_v1.py 复制的核心函数 ===

def init_generation_model(base_model_path: str, lora_adapter_path: Optional[str] = None):
    """
    加载 LLM 和 Tokenizer，并包装成 pipeline。
    与你的 rag_manual_v1.py 中的函数相同。
    """
    # 加载基础模型
    dtype = torch.float16 if DEVICE.startswith("cuda") else torch.float32
    logger.info("加载 base 模型 (dtype=%s, device=%s)...", dtype, DEVICE)
    
    # 检查路径是否存在
    if not os.path.exists(base_model_path):
        logger.error("!!! 致命错误: 找不到 Base 模型路径: %s", base_model_path)
        st.error(f"错误: 找不到 Base 模型路径: {base_model_path}")
        st.stop()
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    )
    
    # 如有 LoRA adapter，加载
    if lora_adapter_path and os.path.exists(lora_adapter_path):
        try:
            model = PeftModel.from_pretrained(base_model, lora_adapter_path)
            logger.info("LoRA adapter 从 %s 加载成功", lora_adapter_path)
        except Exception as e:
            logger.warning("加载 LoRA 失败 (%s)，将使用 base model。", e)
            model = base_model
    elif lora_adapter_path:
        logger.warning("提供了 LoRA 路径但未找到: %s，将使用 base model。", lora_adapter_path)
        model = base_model
    else:
        logger.info("未提供 LoRA 路径，使用 base model。")
        model = base_model
        
    model.use_cache = False
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, use_fast=False)
    
    # wrap pipeline
    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=dtype,
        max_new_tokens=512,  # 默认值
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
    简单 prompt 拼接策略。
    与你的 rag_manual_v1.py 中的函数相同。
    """
    ctxs = []
    for i, d in enumerate(docs, 1):
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
    """
    执行 RAG 流程（检索、构建Prompt、生成）。
    与你的 rag_manual_v1.py 中的函数相同。
    """
    # 1. 检索
    logger.info("正在为问题检索: %s", question[:50] + "...")
    try:
        retrieved = vector_db.similarity_search(question, k=k)
        logger.info("检索到 %d 条文档", len(retrieved))
    except Exception as e:
        logger.error("FAISS 检索失败: %s", e)
        return {"answer": f"抱歉，检索文档时出错: {e}", "source_documents": []}

    # 2. 构建 prompt
    prompt = compose_prompt_with_context(question, retrieved)

    # 3. 生成
    logger.info("开始生成答案...")
    try:
        out = gen_pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9)
        raw_text = out[0].get("generated_text", "")
        
        # 去掉 prompt 部分
        if raw_text.startswith(prompt):
            answer = raw_text[len(prompt):].strip()
        else:
            answer = raw_text.strip()
            
        logger.info("答案生成完毕。")
    except Exception as e:
        logger.error("模型生成失败: %s", e)
        answer = f"抱歉，模型生成答案时出错: {e}"

    # 4. 返回 answer + sources
    result = {
        "answer": answer,
        "source_documents": retrieved # 返回原始 Document 对象
    }
    return result

# === Streamlit 缓存加载函数 ===

@st.cache_resource
def load_faiss_index(_emb_model_path, _index_dir):
    """
    (仅 Streamlit) 缓存加载 FAISS 索引和嵌入模型。
    """
    logger.info("--- 正在加载 FAISS 索引和 BGE 嵌入模型 ---")
    if not os.path.exists(_emb_model_path):
        logger.error("!!! 致命错误: 找不到嵌入模型路径: %s", _emb_model_path)
        st.error(f"错误: 找不到嵌入模型路径: {_emb_model_path}")
        st.stop()
    if not os.path.exists(_index_dir):
        logger.error("!!! 致命错误: 找不到 FAISS 索引: %s", _index_dir)
        st.error(f"错误: 找不到 FAISS 索引目录: {_index_dir}")
        st.info("请先运行 `build_index.py` 脚本来创建索引。")
        st.stop()
        
    try:
        emb = HuggingFaceEmbeddings(
            model_name=_emb_model_path,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )
        vector_db = FAISS.load_local(_index_dir, emb, allow_dangerous_deserialization=True)
        logger.info("--- FAISS 索引和 BGE 加载完成 ---")
        return vector_db
    except Exception as e:
        logger.error("!!! 致命错误: 加载 FAISS 索引失败: %s", e)
        st.error(f"加载 FAISS 索引失败: {e}")
        st.stop()


@st.cache_resource
def load_llm_pipeline(_base_model_path, _lora_adapter_path):
    """
    (仅 Streamlit) 缓存加载 LLM 生成 pipeline。
    """
    logger.info("--- 正在加载 LLM (Baichuan2 + LoRA) ---")
    # 复用你的函数
    _, _, gen_pipe = init_generation_model(_base_model_path, _lora_adapter_path)
    logger.info("--- LLM Pipeline 加载完成 ---")
    return gen_pipe

# === Streamlit UI 界面 ===

# --- 页面配置 ---
st.set_page_config(
    page_title="法律大模型 RAG 问答",
    page_icon="⚖️",
    initial_sidebar_state="collapsed"
)

# --- CSS 样式 (来自你的 minimind 示例) ---
st.markdown("""
    <style>
        /* (这里省略了你提供的长串 CSS，保持原样) */
        /* ... 你提供的所有 .stButton, .stMainBlockContainer, .stApp 样式 ... */
        
        /* 聊天消息样式 */
        .stChatMessage {
            border-radius: 10px;
            padding: 12px;
            margin-bottom: 10px;
        }
        /* 用户消息 (靠右) */
        div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][style*="flex-end"]) {
            /* 也许添加一个背景色 */
        }
        /* 助手消息 (靠左) */
        div[data-testid="chat-message-container"]:has(div[data-testid="stChatMessageContent"][style*="flex-start"]) {
            /* 也许添加一个背景色 */
        }
        
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

# --- 标题和Slogan ---
st.markdown(
    f'<div style="display: flex; flex-direction: column; align-items: center; text-align: center; margin: 0; padding: 0;">'
    '<div style="font-style: italic; font-weight: 900; margin: 0; padding-top: 4px; display: flex; align-items: center; justify-content: center; flex-wrap: wrap; width: 100%;">'
    f'<span style="font-size: 40px; margin-right: 10px;">⚖️</span>'
    f'<span style="font-size: 26px; margin-left: 10px;">法律大模型 RAG 问答</span>'
    '</div>'
    '<span style="color: #bbb; font-style: italic; margin-top: 6px; margin-bottom: 10px;">内容由AI生成，并基于检索的法条，请仔细甄别</span>'
    '</div>',
    unsafe_allow_html=True
)

# --- 加载模型和索引 ---
# (这会在页面首次加载时运行，并缓存结果)
try:
    vector_db = load_faiss_index(EMB_MODEL_PATH, FAISS_INDEX_DIR)
    gen_pipe = load_llm_pipeline(BASE_MODEL_PATH, LORA_ADAPTER_PATH)
    st.success("模型和索引加载成功！", icon="✅")
except Exception as e:
    # 错误已在加载函数内部处理 (st.error + st.stop)
    pass


# --- 聊天界面 ---

# 初始化 session_state
if "messages" not in st.session_state:
    st.session_state.messages = []

# 在侧边栏添加清空按钮
st.sidebar.title("选项")
if st.sidebar.button("清空聊天记录", use_container_width=True):
    st.session_state.messages = []
    st.rerun() # 重新运行以清空界面

# 显示历史消息
for message in st.session_state.messages:
    avatar = "⚖️" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        # 助手消息可能包含 HTML (用于来源)
        if message["role"] == "assistant":
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# 获取用户输入
if prompt := st.chat_input("请输入你的法律问题..."):
    # 1. 将用户消息添加到 session_state
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # 3. 生成助手回复
    with st.chat_message("assistant", avatar="⚖️"):
        # 使用占位符显示 "思考中..."
        placeholder = st.empty()
        placeholder.markdown("🔍 正在检索相关法条并生成答案...")
        
        # 调用你的 RAG 核心函数
        # (这是阻塞操作，会等待模型返回)
        result_dict = answer_by_rag(gen_pipe, vector_db, prompt, k=TOP_K)
        
        answer = result_dict.get("answer", "抱歉，未能生成答案。")
        sources = result_dict.get("source_documents", [])
        
        # 4. 格式化并显示助手答案
        full_response = f"{answer}\n\n"
        
        if sources:
            full_response += '<div class="source-container"><strong>参考来源：</strong>\n'
            for i, doc in enumerate(sources, 1):
                # 提取元数据和内容片段
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
            full_response += '<div class="source-container"><strong>未检索到相关上下文。</strong></div>'

        # 更新占位符
        placeholder.markdown(full_response, unsafe_allow_html=True)
        
        # 5. 将助手消息添加到 session_state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

#streamlit run app.py --server.address=127.0.0.1 --server.port=6006
