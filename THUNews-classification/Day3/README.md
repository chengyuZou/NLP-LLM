封装 Dataset：直接用 datasets 加载 JSONL，按需重命名/过滤字段。

自定义 DataCollator：继承或编写一个可对齐 input_ids、生成 attention_mask、打包 labels 的函数/类。

可视化：用 Matplotlib 分别绘制样本长度的直方图和 Token 频次的柱状图，帮助你直观了解数据分布。
