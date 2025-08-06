流程详解

Warmup + Cosine Decay
warmup_steps=500：先线性升到初始 learning_rate。
get_cosine_schedule_with_warmup：后续按照余弦曲线缓慢衰减至 0。

Early Stopping
EarlyStoppingCallback(early_stopping_patience=3)：验证集上 F1 连续 3 次不提升，则提前结束训练。

TensorBoard 可视化
report_to="tensorboard" 与 logging_dir="./runs"：Trainer 会自动每隔 logging_steps 向 TensorBoard 写入 Loss、LR、Metrics。
在 TensorBoard 的 Scalars 面板，你可以看到：
train/loss：训练损失曲线
eval/loss：验证损失曲线
learning_rate：LR 随步数变化曲线
eval/accuracy、eval/f1：验证指标

output_dir文件太大，约30G，故没放上去
