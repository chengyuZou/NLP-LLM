import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datetime
import os


# 初始化进程组
dist.init_process_group(backend='nccl', init_method='env://')

# 设置本地设备
local_rank = int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(local_rank)
device = torch.device("cuda")

# 创建模型
model = ...  # 替换为你的模型
device = torch.device("cuda:{}".format(rank))
model.to(device)

# 重点一：包装数据集
# load training data
train_data = Dataset(...)
train_sampler = DistributedSampler(train_data)
training_data_loader = DataLoader(dataset=train_data, batch_size=batch_size, sampler=train_sampler, drop_last=True, num_workers=opt.num_workers)

# 重点二：包装模型
ddp_model = DDP(model, device_ids=[rank])

# 定义损失函数和优化器
criterion = ...
optimizer = ...

# 训练循环
for epoch in range(num_epochs):
	# 重点三：设置 sampler 的 epoch，DistributedSampler 需要这个来维持各个进程之间的随机种子，也就是保证所有进程在数据洗牌时使用的随机种子是一致的，这样每个进程就会得到不同的数据子集，但整个训练集上的采样是一致的，
	train_sampler.sampler.set_epoch(epoch)
    for iteration, data in enumerate(training_data_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # DDP将在这里同步梯度
        optimizer.step()

        # 日志和其他操作
        if rank == 0:
            # 打印日志或写入日志文件
            ...

# 清理
dist.destroy_process_group()

