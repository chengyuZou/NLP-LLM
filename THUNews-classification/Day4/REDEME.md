运行说明:

准备数据
将 Day 1 抽取好的子集拆分为训练/验证两份，例如：

# 假设原 jsonl 是 THUCNews.jsonl
split -l 8000 THUCNews.jsonl THUCNews_
mv THUCNews_aa THUCNews_train.jsonl
mv THUCNews_ab THUCNews_val.jsonl

也可用 Python 随机划分，保证类内比例一致。
运行脚本

cd your_path
python Test.py
查看结果

或者直接运行Test的Nodebook

脚本结束后，控制台会打印出验证集上的 Accuracy 和 Macro-F1，模型检查点保存在 ./bert_chinese_baseline。

注意：原json文件的lebel是str类型，要转换为int类型，因此需要一个映射

这样，你就能快速获得一个基于 bert-base-chinese 的下游分类基线，并输出验证准确率和 F1。后续可以在此基础上调学习率、batch size、冻结层数、增加数据增强等，进一步提升性能。
