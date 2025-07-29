## Step 1
数据集下载地址
THUCNews 原始数据集包含 14 个新闻类别、约 74 万篇新闻，推荐以下几种常用镜像：

官方 Figshare 链接（包含所有 14 类，约 1.45 GB，UTF‑8 文本）：
https://figshare.com/articles/dataset/THUCNews_Chinese_News_Text_Classification_Dataset/28279964 
Figshare

Kaggle 镜像（需要注册并登录 Kaggle）：
https://www.kaggle.com/datasets/cycloneboy/thucnews 
Kaggle

Hugging Face Datasets（需 datasets 库加载，支持直接 .to_parquet() 转换）：
seamew/THUCNews 
Hugging Face

## Step 2
子集抽取与清洗流程
假设你已经把原始数据解压到本地目录 THUCNews/，结构如下：

THUCNews/
├── finance/
│   ├── 0001.txt
│   ├── 0002.txt
│   └── ...
├── sports/
├── technology/
└── ...（共 14 类）

你想做的是：
抽取 5 个类别（比如：财经、教育、科技、娱乐、体育）
每类 2000 条
去空、去重
保存成标准 JSONL 格式，用于后续模型训练

import os
import glob
import pandas as pd

# 1. 设置参数
data_dir = "THUCNews1"  # 改成你实际路径
selected_labels = ["财经", "教育", "科技", "娱乐", "体育"]
samples_per_class = 2000
random_seed = 42

# 2. 读取并清洗所有样本
records = []

for label in selected_labels:
    class_path = os.path.join(data_dir, label)
    txt_files = glob.glob(os.path.join(class_path, "*.txt"))
    
    print(f"正在读取类别 {label}，共 {len(txt_files)} 条")

    for fp in txt_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:  # 丢弃空文本
                records.append({"text": text, "label": label})
        except Exception as e:
            print(f"读取失败：{fp}，错误：{e}")

## Step 3. 构建 DataFrame 并去重
df = pd.DataFrame(records)
print("原始总数量：", len(df))
df.drop_duplicates(subset=["text"], inplace=True)
df.dropna(subset=["text", "label"], inplace=True)
print("去重 & 去空后：", len(df))

## Step 4. 按类别均匀抽样
df_sampled = (
    df.groupby("label", group_keys=False)
      .apply(lambda x: x.sample(n=samples_per_class, random_state=random_seed))
      .reset_index(drop=True)
)
print("采样后总数量：", len(df_sampled))

# 5. 保存为 JSON Lines 格式
output_file = "THUCNews5类2000条.jsonl"
df_sampled.to_json(output_file, orient="records", lines=True, force_ascii=False)
print(f"已保存为 {output_file}")


✅ 输出文件结构（JSONL 样例）
保存后你会得到类似这样的内容：

json
{"text": "阿里巴巴宣布新一轮融资计划......", "label": "财经"}
{"text": "教育部发布最新政策，强调素质教育......", "label": "教育"}
{"text": "腾讯推出全新AI大模型......", "label": "科技"}
{"text": "某演员出演新电影......", "label": "娱乐"}
{"text": "世界杯预选赛中国队对阵韩国队......", "label": "体育"}
...


