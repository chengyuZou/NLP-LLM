random.choice(seq)
📌 作用：从一个序列中随机选一个元素
import random
items = ['apple', 'banana', 'cherry']
print(random.choice(items))  # 输出可能是 'banana'

✅ random.randint(a, b)
📌 作用：生成一个**[a, b]（闭区间）之间的随机整数**
print(random.randint(1, 10))  # 输出 1~10 之间的整数

✅ random.sample(seq, k)
📌 作用：从序列中随机抽取 k 个不重复元素
nums = [1, 2, 3, 4, 5]
print(random.sample(nums, 3))  # 例如 [2, 5, 1]

✅ random.random()
📌 作用：生成一个 0 到 1 之间的浮点数

(未尝试)另外检查 eda_augment() 中是否是按空格切词
EDA 代码通常假设输入是英文词列表，通过空格分词：
words = sentence.split()
但对于中文，你需要使用更合适的分词器，比如 jieba：
import jieba
words = list(jieba.cut(sentence))
否则 "糖尿病患者需要监测血糖水平" 会被认为是一个词而非多个词，结果 len(words) == 1，就会触发上面这个错误。

1. 安装依赖
pip install nltk
python -m nltk.downloader wordnet omw-1.4
说明
我们用 NLTK 的 WordNet 来做同义词替换；EDA 的四种操作（插入、删除、替换、交换）

2. Easy Data Augmentation (EDA)
参考 Wei & Zou (2019) 的 EDA 四种操作：
随机删除 (Random Deletion)
随机交换 (Random Swap)
随机插入 (Random Insertion)
随机替换 (Random Replacement; 类似同义词替换但概率更高)
