import json
import torch
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.llms.base import LLM
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
import logging
from typing import Optional, List, Mapping, Any

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 自定义 LangChain 兼容的 LLM 类
class BaichuanLLM(LLM):
    """自定义 Baichuan LLM 包装器"""
    
    model: Any
    tokenizer: Any
    pipeline: Any
    
    def __init__(self, model, tokenizer, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.tokenizer = tokenizer
        # 创建 transformers pipeline
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            # 使用 pipeline 生成文本
            result = self.pipeline(
                prompt,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1
            )
            return result[0]['generated_text'][len(prompt):].strip()
        except Exception as e:
            logger.error(f"生成文本时发生错误: {e}")
            return f"生成回答时出错: {str(e)}"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"name": "Baichuan2-7B-LoRA-Legal"}
    
    @property
    def _llm_type(self) -> str:
        return "baichuan_legal_qa"

# 加载数据
def load_qa_data(file_path: str) -> List[Document]:
    """
    加载QA数据并转换为Document格式
    
    Args:
        file_path: QA数据文件路径
        
    Returns:
        Document列表
    """
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    page_content = f"问题：{data['input']}\n答案：{data['output']}"
                    metadata = {
                        "id": data.get("id", f"line_{line_num}"), 
                        "source": "legal_qa",
                        "line_number": line_num
                    }
                    documents.append(Document(page_content=page_content, metadata=metadata))
                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_num}行JSON解析错误: {e}")
                    continue
                except KeyError as e:
                    logger.warning(f"第{line_num}行缺少必要字段: {e}")
                    continue
                    
        logger.info(f"成功加载 {len(documents)} 条QA数据")
        return documents
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    except Exception as e:
        logger.error(f"加载数据时发生错误: {e}")
        raise

def create_chunks(documents: List[Document], chunk_size: int = 256, chunk_overlap: int = 50) -> List[Document]:
    """
    将文档分块
    
    Args:
        documents: Document列表
        chunk_size: 块大小
        chunk_overlap: 重叠大小
        
    Returns:
        分块后的Document列表
    """
    # 使用适合中文法律文本的分隔符
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"原文档数: {len(documents)}, 分块后: {len(chunks)}")
    return chunks

def create_vector_store(chunks: List[Document], model_path: str = "./bge-large-zh-v1.5/") -> FAISS:
    """
    创建向量存储
    
    Args:
        chunks: 文档块列表
        model_path: 嵌入模型路径
        
    Returns:
        FAISS向量存储
    """
    try:
        # 构建嵌入模型
        embedding_model = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={'normalize_embeddings': True,
                          'batch_size': 64}
        )
        
        # 创建向量数据库
        vector_db = FAISS.from_documents(chunks, embedding_model)
        logger.info("向量数据库创建成功")
        return vector_db, embedding_model
    except Exception as e:
        logger.error(f"创建向量数据库时发生错误: {e}")
        raise

def save_vector_store(vector_db: FAISS, path: str = "faiss_legal_qa_index"):
    """
    保存向量存储到本地
    
    Args:
        vector_db: FAISS向量存储
        path: 保存路径
    """
    try:
        vector_db.save_local(path)
        logger.info(f"向量数据库已保存到: {path}")
    except Exception as e:
        logger.error(f"保存向量数据库时发生错误: {e}")
        raise

def load_vector_store(path: str, embedding_model: HuggingFaceEmbeddings) -> FAISS:
    """
    从本地加载向量存储
    
    Args:
        path: 加载路径
        embedding_model: 嵌入模型
        
    Returns:
        FAISS向量存储
    """
    try:
        vector_db = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)
        logger.info(f"向量数据库已从 {path} 加载")
        return vector_db
    except Exception as e:
        logger.error(f"加载向量数据库时发生错误: {e}")
        raise

def search_similar_documents(vector_db: FAISS, query: str, k: int = 10) -> List[Document]:
    """
    搜索相似文档
    
    Args:
        vector_db: FAISS向量存储
        query: 查询文本
        k: 返回结果数量
        
    Returns:
        相似文档列表
    """
    try:
        similar_docs = vector_db.similarity_search(query, k=k)
        logger.info(f"查询 '{query}' 找到 {len(similar_docs)} 个相似文档")
        return similar_docs
    except Exception as e:
        logger.error(f"相似性搜索时发生错误: {e}")
        raise

def initialize_model(base_model_path: str, lora_adapter_path: str):
    """
    初始化模型
    
    Args:
        base_model_path: 基础模型路径
        lora_adapter_path: LoRA适配器路径
        
    Returns:
        模型和tokenizer
    """
    try:
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA微调模型
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        
        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        
        logger.info("模型加载成功")
        return model, tokenizer
    except Exception as e:
        logger.error(f"模型加载时发生错误: {e}")
        raise

def create_qa_chain(model, tokenizer, vector_db, k: int = 3):
    """
    创建问答链
    
    Args:
        model: 模型
        tokenizer: Tokenizer
        vector_db: 向量数据库
        k: 检索文档数量
        
    Returns:
        QA链
    """
    try:
        from langchain.llms import HuggingFacePipeline
        
        # 创建 transformers pipeline
        text_generation_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # 创建 LangChain LLM
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
        # 创建检索与问答链
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_db.as_retriever(search_kwargs={"k": k}),
            return_source_documents=True
        )
        logger.info("问答链创建成功")
        return qa_chain
    except Exception as e:
        logger.error(f"创建问答链时发生错误: {e}")
        raise

def main():
    """主函数"""
    # 配置参数
    data_file_path = "/root/autodl-tmp/SFT/LoRA/LoRA_data.jsonl"
    embedding_model_path = "./bge-large-zh-v1.5/"
    base_model_path = "./Baichuan2-7B-Base/"
    lora_adapter_path = "./lora_legal_qa_adapter"
    faiss_index_path = "faiss_legal_qa_index"
    
    try:
        # 1. 加载数据
        #documents = load_qa_data(data_file_path)
        
        # 2. 文档分块
        #chunks = create_chunks(documents, chunk_size=1024, chunk_overlap=50)
        
        # 3. 创建向量存储
        #vector_db, embedding_model = create_vector_store(chunks, embedding_model_path)
        
        # 4. 保存向量存储
        #save_vector_store(vector_db, faiss_index_path)
        embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_path,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={'normalize_embeddings': True,
                          'batch_size': 64}
        )

        vector_db = load_vector_store(faiss_index_path , embedding_model)
        
        # 5. 相似性搜索示例
        query = "劳动合同法保护哪些权益？"
        similar_docs = search_similar_documents(vector_db, query, k=10)
        
        print(f"针对查询：\n{query}")
        print("找到的相似文档：")
        for idx, doc in enumerate(similar_docs):
            print(f"{idx + 1}: {doc.page_content}")
            print(f"元数据：{doc.metadata}\n")
        
        # 6. 初始化模型
        model, tokenizer = initialize_model(base_model_path, lora_adapter_path)
        
        # 7. 创建问答链
        qa_chain = create_qa_chain(model, tokenizer, vector_db, k=3)
        
        # 8. 生成答案
        result = qa_chain.invoke({"query": query})
        print(f"生成的答案：\n{result['result']}")
        
        # 显示源文档
        print("\n参考的源文档：")
        for idx, doc in enumerate(result['source_documents']):
            print(f"文档 {idx + 1}: {doc.page_content}")
            
    except Exception as e:
        logger.error(f"执行过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()