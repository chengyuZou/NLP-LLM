# **⚖️ 基于 Baichuan2-7B 的法律微调大模型与 RAG 系统**

本项目实现了一个端到端的中文法律问答系统。它使用 LoRA 对 Baichuan2-7B 模型在法律 QA 数据集上进行了两次微调，并结合了高级 RAG（检索增强生成）流程实现深度搜索功能，提供了一个可交互的 Streamlit Web 界面。


## **演示效果**

https://github.com/user-attachments/assets/81f689de-515e-493b-ba10-71d01b718289

## **需求**
* **平台**: AutoDL
* **显卡**: A800 1张(80G显存)
* **镜像**:
```text
PyTorch == 2.8.0
Python == 3.12(ubuntu22.04)
CUDA == 12.8
```

## **核心技术栈**

* **模型**: baichuan-inc/Baichuan2-7B-Base  
* **微调**: PEFT (LoRA), Transformers Trainer  
* **嵌入 (Embedding)**: bge-large-zh-v1.5  
* **重排 (Reranker)**: bge-rerank-large  
* **检索 (Retrieval)**: rank-bm25 (稀疏), faiss (稠密)  
* **框架**: Pytorch ,LangChain, Streamlit  
* **评测**: Perplexity (PPL), Rouge , HR, MRR
* **RAG 流程**: BM25 (稀疏检索) \+ FAISS/BGE (稠密检索) \-\> RRF (多路召回融合) \-\> BGE-Reranker (精排) \-\> LLM (生成答案)


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

1. **基础模型**: 从 [Hugging_Face](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)下载 Baichuan2-7B-Base 并放入 models/ 目录。  
2. **嵌入模型**: 从 [Hugging_Face](https://huggingface.co/BAAI/bge-large-zh-v1.5)下载 bge-large-zh-v1.5 并放入 models/ 目录。
3. **Rerank模型** 从 [Hugging_Face](https://huggingface.co/BAAI/bge-reranker-large)下载 BAAI/bge-reranker-large 并放入 models/目录
4. **本项目 LoRA 适配器**: 从 [Hugging Face](https://huggingface.co/erfsdfds/BaiChuan2-7B-Law-SFT) 下载我的 两个LoRA 权重，并放入 models 目录里解压。 解压后分别命名为 lora_legal_qa_adapter 和 lora_new_legal_qa_adapter
5. **数据集**: 从 [Hugging Face](https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT) 下载两个数据集 DISC-Law-SFT-Pair-QA-released.jsonl与DISC-Law-SFT-Triplet-QA-released.jsonl 并放入 data/ 目录。

*(请在 src/config.py 中检查并确认所有路径均正确配置)*

### **1.3. 构建 RAG 索引**

在运行 UI 之前，你需要先为 RAG 系统构建向量数据库和 BM25 索引。

\# 这将读取 data/ 中的 jsonl 文件，并创建索引到 index/ 目录  
```bash
python scripts/build_index.py
```

### **1.4. 启动 Streamlit 应用**

```bash
# 启动 Web UI  
streamlit run app/app.py --server.address=127.0.0.1 --server.port=6006
```

现在你可以打开浏览器访问 http://127.0.0.1:6006 开始提问。

## **2\. 进阶：复现项目（训练与评测）**

如果你想从头开始复现整个项目，请按顺序执行 scripts/ 目录中的脚本。顺序为:
1. prepare_data.py

2. train_lora_sft.py

3. triann_new_lora.py

4. evaluate_ppl.py

5. build_index.py

6. evaluate_rag.py

### **2.1. 准备 SFT 数据**

此脚本将原始的 Pair-QA 数据转换为 LoRA\_data.jsonl 格式。

```bash
python scripts/prepare_data.py
```

### **2.2. 训练 LoRA 模型**

此脚本将加载 Base 模型和 LoRA\_data.jsonl，执行 SFT，并将新的 LoRA 适配器保存到 models/ 目录。

\# 确保在 src/config.py 中配置了正确的路径

```bash
torchrun --nproc_per_node=x train_lora_sft.py
```

其中(x)为你的显卡数量. 有几张显卡，这里就填几

eg: 
```bash
torchrun --nproc_per_node=2 train_lora_sft.py # 运行两个显卡
```

### **2.3. 训练新 LoRA 模型**

此脚本将加载 第一次训练的LoRA 模型和 data/DISC-Law-SFT-Triplet-QA-released.jsonl 执行 SFT，并将新的 LoRA 适配器保存到 models/ 目录。

同理

```bash
torchrun --nproc_per_node=x triann_new_lora.py
```

### **2.4. 评测模型（困惑度 PPL）**

此脚本将分别计算 Base 模型和 LoRA 模型在测试集上的困惑度（Perplexity）。

```bash
python scripts/evaluate_ppl.py
```

#### **评测结果 (PPL)**

| 模型 | Mean Perplexity | Mean Loss | PPL 提升 |
| :---- | :---- | :---- | :---- |
| Baichuan2-7B (Base) | 4.4633 | 1.4959 | \- |
| **Baichuan2-7B (LoRA v2)** | **3.7871** | **1.3316** | **15.15%** |

### **2.5. 构建索引**

此脚本将读取 data/DISC-Law-SFT-Pair-QA-released.jsonl 文件，并创建索引到 index/ 目录,以便检索数据。

```bash
python scripts/build_index.py
```

### **2.6. 评测 RAG 系统**

此脚本将全面评估 RAG pipeline 的质量。

```bash
python scripts/evaluate_rag.py
```

## **项目路线图 (Roadmap)**

本项目还在持续改进中，后续计划包括：
* \[ x\] **联网搜索**: 有深度思考怎么能没有联网搜索？
* \[ x\] **代码详解**: 每个代码都有一个讲解的markdown 文件。
* \[ x\] **RAG 评测**: 使用 RAGAS 代替 HitRate/MRR (待完成)  
* \[ x\] **DPO 优化**: 解决模型对话生硬问题，在 SFT 基础上进行 DPO。  
* \[ x\] **纯手写**: 不使用各种库函数,手写实现各种功能

## **许可证 (License)**

本项目采用 [MIT License](https://www.google.com/search?q=LICENSE) 许可。

