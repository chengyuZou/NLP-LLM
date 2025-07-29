一、子词模型原理概述
Byte-Pair Encoding (BPE)

基于“最频对合并”思想，从字符级开始，迭代地将出现频率最高的相邻字符对合并为一个新符号。
优点：训练速度快、易实现、对低频词有较好分割；
缺点：对未知字符无内置概率，分割较“刚性”。

SentencePiece（Unigram）

基于概率模型（Unigram language model），先从大词表抽取候选子词，再通过 EM 算法学习每个子词的概率，删减概率较低的子词。
优点：分割更灵活，可处理噪音字符；
缺点：训练相对慢，对大词表敏感。

中英文混合

中文无需空格分词，直接按字符或 n-gram 统计；英文按普通 BPE/Unigram 处理。
建议先做“预归一化”：所有字母转小写、保留数字/英文符号，中文字符按 Unicode 保留。

二、获得训练文件
先创建train_bpe/unigram.py文件

实验编号	模型	vocab_size	其他超参
1	BPE	8 k	char_coverage=0.9995
2	BPE	16 k	char_coverage=0.9995
3	BPE	32 k	char_coverage=0.9995
4	Unigram	8 k	unk_surface=”<unk>”
5	Unigram	16 k	shrink_factor=0.1
6	Unigram	32 k	shrink_factor=0.2

vocab_size：影响 OOV 率和序列长度，通常 8k–32k 是中英混合任务常见选择。
char_coverage（仅 SentencePiece）：指定训练时覆盖大部分字符频次，防止罕见字符被忽略。
shrink_factor（仅 Unigram）：模型迭代剪枝比例。

三、vocab-size 对比实验
创建train_classifier.py Test.py 与 eval_experiments.py文件，在终端进行训练，最终将不同 size 下的准确率、序列长度、OOV 率整理到表格，可用 Matplotlib（单图）绘制折线对比。
