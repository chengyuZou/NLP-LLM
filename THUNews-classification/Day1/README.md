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
    
    data_dir = "THUCNews1"
    selected_labels = ["财经", "教育", "科技", "娱乐", "体育"]
    samples_per_class = 2000
    random_seed = 42
    
    records = []
    
    for label in selected_labels:
        class_path = os.path.join(data_dir, label)
        txt_files = glob.glob(os.path.join(class_path, "*.txt"))

    print(f"读取 {label} 类，共 {len(txt_files)} 文件")

    for fp in txt_files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                records.append({"text": text, "label": label})
        except Exception as e:
            print(f"读取失败：{fp}，错误：{e}")

    print("总记录数：", len(records))
    print("前几条：", records[:2])
    
    # 如果 records 是空的，就报错退出
    if not records:
        raise ValueError("❌ 没有读取到任何有效文本，请检查路径是否正确，以及是否是 UTF-8 编码")
    
    df = pd.DataFrame(records)
    print("DataFrame 列名：", df.columns.tolist())
    
    # 去空、去重
    df.dropna(subset=["text"], inplace=True)
    df.drop_duplicates(subset=["text"], inplace=True)
    
    # 每类采样
    df_sampled = (
        df.groupby("label", group_keys=False)
          .apply(lambda x: x.sample(n=samples_per_class, random_state=random_seed))
          .reset_index(drop=True)
    )
    
    # 保存
    df_sampled.to_json("THUCNews5类2000条.jsonl", orient="records", lines=True, force_ascii=False)
    print("✅ 已保存为 JSONL 格式")


✅ 输出文件结构（JSONL 样例）
保存后你会得到类似这样的内容：

json
{"text": "阿里巴巴宣布新一轮融资计划......", "label": "财经"}
{"text": "教育部发布最新政策，强调素质教育......", "label": "教育"}
{"text": "腾讯推出全新AI大模型......", "label": "科技"}
{"text": "某演员出演新电影......", "label": "娱乐"}
{"text": "世界杯预选赛中国队对阵韩国队......", "label": "体育"}
...


