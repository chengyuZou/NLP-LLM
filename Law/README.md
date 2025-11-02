# **⚖️ 基于 Baichuan2-7B 的法律微调大模型与 RAG 系统**

本项目实现了一个端到端的中文法律问答系统。它使用 LoRA 对 Baichuan2-7B 模型在法律 QA 数据集上进行了两次微调，并结合了高级 RAG（检索增强生成）流程，提供了一个可交互的 Streamlit Web 界面。

**RAG 流程**: BM25 (稀疏检索) \+ FAISS/BGE (稠密检索) \-\> RRF (多路召回融合) \-\> BGE-Reranker (精排) \-\> LLM (生成答案)

## **演示效果**

![0896411bb7d4bdae1c0658353037e888](https://github.com/user-attachments/assets/73f3ec0c-c5c5-4512-9b78-2538b031a1b8)


## **核心技术栈**

* **模型**: baichuan-inc/Baichuan2-7B-Base  
* **微调**: PEFT (LoRA), Transformers Trainer  
* **嵌入 (Embedding)**: bge-large-zh-v1.5  
* **重排 (Reranker)**: bge-rerank-large  
* **检索 (Retrieval)**: rank-bm25 (稀疏), faiss (稠密)  
* **框架**: Pytorch ,LangChain, Streamlit  
* **评测**: Perplexity (PPL), Rouge , HR, MRR

## **1\. 快速开始：运行 UI 界面**

按照以下步骤在本地启动 RAG 问答界面。

### **1.1. 环境配置**

\# 1\. 克隆仓库  
```bash
git clone https://github.com/chengyuZou/NLP-LLM.git  
cd NLP-LLM
cd Law
```

\# 2\. 安装依赖  
```bash
pip install -r requirements.txt
```

### **1.2. 下载模型和数据**

本项目需要以下预训练模型和数据：

1. **基础模型**: 从 [Hugging_Face](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)下载 Baichuan2-7B-Base 并放入 models/Baichuan2-7B-Base/ 目录。  
2. **嵌入模型**: 从 [Hugging_Face](https://huggingface.co/BAAI/bge-large-zh-v1.5)下载 bge-large-zh-v1.5 并放入 models/bge-large-zh-v1.5/ 目录。
3. **Rerank模型** 从 [Hugging_Face](https://huggingface.co/BAAI/bge-reranker-large)下载 BAAI/bge-reranker-large 并放入 models/bge-rerank-large/目录
4. **本项目 LoRA 适配器**: 从 [Hugging Face](https://huggingface.co/erfsdfds/BaiChuan2-7B-Law-SFT) 下载我的 两个LoRA 权重，并放入 models 目录里解压。 解压后分别命名为 lora_legal_qa_adapter 和 lora_new_legal_qa_adapter
5. **数据集**: 从 [Hugging Face](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) 下载两个数据集 DISC-Law-SFT-Pair-QA-released.jsonl与DISC-Law-SFT-Triplet-QA-released.jsonl 并放入 data/ 目录。

*(请在 src/config.py 中检查并确认所有路径均正确配置)*

### **1.3. 构建 RAG 索引**

在运行 UI 之前，你需要先为 RAG 系统构建向量数据库和 BM25 索引。

\# 这将读取 data/ 中的 jsonl 文件，并创建索引到 index/ 目录  
python scripts/build\_index.py

### **1.4. 启动 Streamlit 应用**

\# 启动 Web UI  
streamlit run app/app.py \--server.address=127.0.0.1 \--server.port=6006

现在你可以打开浏览器访问 http://127.0.0.1:6006 开始提问。

## **2\. 进阶：复现项目（训练与评测）**

如果你想从头开始复现整个项目，请按顺序执行 scripts/ 目录中的脚本。顺序为:
1. prepare\_data.py

2. train\_lora\_sft.py

3. triann\_new\_lora.py

4. evaluate\_ppl.py

5. build\_index.py

6. evaluate\_rag.py

### **2.1. 准备 SFT 数据**

此脚本将原始的 Pair-QA 数据转换为 LoRA\_data.jsonl 格式。

python scripts/1\_prepare\_data.py

### **2.2. 训练 LoRA 模型**

此脚本将加载 Base 模型和 LoRA\_data.jsonl，执行 SFT，并将新的 LoRA 适配器保存到 models/ 目录。

\# 确保在 src/config.py 中配置了正确的路径  
python scripts/train\_lora\_sft.py

### **2.3. 评测模型（困惑度 PPL）**

此脚本将分别计算 Base 模型和 LoRA 模型在测试集上的困惑度（Perplexity）。

python scripts/evaluate\_ppl.py

#### **评测结果 (PPL)**

| 模型 | Mean Perplexity | Mean Loss | PPL 提升 |
| :---- | :---- | :---- | :---- |
| Baichuan2-7B (Base) | 4.4633 | 1.4959 | \- |
| **Baichuan2-7B (LoRA v2)** | **3.7871** | **1.3316** | **15.15%** |

### **2.4. 评测 RAG 系统**

此脚本将全面评估 RAG pipeline 的质量。
 
python scripts/evaluate\_rag.py


## **项目路线图 (Roadmap)**

本项目还在持续改进中，后续计划包括：
* \[ x\] **打注释**: 注释没时间打
* \[ x\] **重构架构**: 将所有代码从 README 迁移到 src/ 和 scripts/ (正在做)  
* \[ x\] **代码详解**: 每个代码都有一个讲解的markdown 文件。
* \[ x\] **RAG 评测**: 使用 RAGAS 代替 HitRate/MRR (待完成)  
* \[ x\] **DPO 优化**: 解决模型对话生硬问题，在 SFT 基础上进行 DPO。  
* \[ x\] **数据清洗**: 对 QA 数据进行去重和基于困惑度的筛选。  
* \[ x\] **多卡训练**: 解决多卡训练（DDP）的配置问题。

## **许可证 (License)**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 许可。

