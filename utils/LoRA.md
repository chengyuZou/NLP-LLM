## 1.LoRA算法思想
**[论文地址](https://arxiv.org/pdf/2106.09685)**  
模型是过参数化的，它们有更小的内在维度，模型主要依赖于这个低的内在维度（low intrinsic dimension）去做任务适配。  
假设模型在适配任务时参数的改变量是低秩的，由此引出低秩自适应方法lora，通过低秩分解来模拟参数的改变量，从而以极小的参数量来实现大模型的间接训练。

## 2.具体实现
LoRA的实现方式是在基础模型的线性变换模块（全连接、Embedding、卷积）旁边增加一个旁路，这个旁路是由两个小矩阵做内积得来的，两个小矩阵的中间维度，就是秩！！  
通过低秩分解（先降维再升维）来模拟参数的更新量。  
下面是LoRA的公式：  
<img width="338" height="21" alt="image" src="https://github.com/user-attachments/assets/6dee4bb2-44e9-466f-82b2-67b8ed4820c1" /> 

上面公式中x是这一层的输入，h是这一层的输出，W_0是基础模型的权重参数；A和B是两个小矩阵，A的输入和B的输出形状跟W_0一样，A的输出和B的输入一样，称为秩。  
秩一般很小，微调的所有“新知识”都保存在A和B里面；\alpha /r是一个缩放系数，这个数越大，LoRA权重的影响就越大。  

LoRA流程图  
<img width="397" height="361" alt="image" src="https://github.com/user-attachments/assets/782335dd-7de2-4822-ac44-d1e7eea09197" />  

## 3.代码实现
### 3.1 LoRA代码主要靠peft库里的LoraConfig, TaskType, get_peft_model函数  
下列给出实例：

```python
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModel, HfArgumentParser, TrainingArguments
 
from finetune import CastOutputToFloat, FinetuneArguments
 
 
def count_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
 
 
 
def make_peft_model():
    # 初始化原模型
    model = AutoModel.from_pretrained(
        "THUDM/chatglm-6b", load_in_8bit=False, trust_remote_code=True, device_map="auto", local_files_only=True
    ).float()
    
 
    # 给原模型施加LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=True,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=['query_key_value'],
    )
    model = get_peft_model(model, peft_config).float()
    count_params(model)
 
 
 
if __name__ == '__main__':
    make_peft_model()
```

输出：  
```python
base_model.model.transformer.word_embeddings.weight torch.Size([130528, 4096])
base_model.model.transformer.layers.0.input_layernorm.weight torch.Size([4096])
base_model.model.transformer.layers.0.input_layernorm.bias torch.Size([4096])
base_model.model.transformer.layers.0.attention.query_key_value.base_layer.weight torch.Size([12288, 4096])
base_model.model.transformer.layers.0.attention.query_key_value.base_layer.bias torch.Size([12288])
base_model.model.transformer.layers.0.attention.query_key_value.lora_A.default.weight torch.Size([8, 4096])
base_model.model.transformer.layers.0.attention.query_key_value.lora_B.default.weight torch.Size([12288, 8])
base_model.model.transformer.layers.0.attention.dense.weight torch.Size([4096, 4096])
base_model.model.transformer.layers.0.attention.dense.bias torch.Size([4096])
base_model.model.transformer.layers.0.post_attention_layernorm.weight torch.Size([4096])
base_model.model.transformer.layers.0.post_attention_layernorm.bias torch.Size([4096])
base_model.model.transformer.layers.0.mlp.dense_h_to_4h.weight torch.Size([16384, 4096])
base_model.model.transformer.layers.0.mlp.dense_h_to_4h.bias torch.Size([16384])
base_model.model.transformer.layers.0.mlp.dense_4h_to_h.weight torch.Size([4096, 16384])
base_model.model.transformer.layers.0.mlp.dense_4h_to_h.bias torch.Size([4096])
base_model.model.transformer.layers.1.input_layernorm.weight torch.Size([4096])
base_model.model.transformer.layers.1.input_layernorm.bias torch.Size([4096])
......
```

### 3.2 我们仿照写一个类似的函数,如下
```python
# 复制粘贴可运行（PyTorch >=1.8）
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # list of substrings to match module names
    inference_mode: bool = False

class LoRALinear(nn.Module):
    """
    Wraps an existing nn.Linear and adds LoRA A,B params.
    - original linear is kept (self.linear)
    - LoRA computes delta = scaling * ( (x @ B.T) @ A.T )
    - merging adds delta into linear.weight.data
    """
    def __init__(self, orig_linear: nn.Linear, r: int, alpha: int=1, dropout: float=0.0):
        super().__init__()
        assert isinstance(orig_linear, nn.Linear)
        self.linear = orig_linear  # keep reference to original linear (weight, bias)
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features

        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = nn.Dropout(p=dropout) if dropout and dropout > 0.0 else None

        if r > 0:
            # LoRA params: A (out x r) and B (r x in)
            # Initialization choices:
            # - A initialized to zeros often ensures initial model behaviour unchanged.
            # - B small random.
            self.A = nn.Parameter(torch.zeros(self.out_features, r))
            self.B = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
        else:
            self.A = None
            self.B = None

        # flag to indicate weights merged into base weight (used for inference)
        self.merged = False

        # freeze original weight by default (we only train A/B)
        # Note: keep bias trainable if you want; here freeze bias too.
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # base output
        base = F.linear(x, self.linear.weight, self.linear.bias)

        if self.r > 0 and not self.merged:
            # LoRA branch: (x @ B.T) -> (batch, r) then @ A.T -> (batch, out)
            if self.dropout:
                x_d = self.dropout(x)
            else:
                x_d = x
            # x_d: (batch, in)
            # B.T: (in, r) -> x_d @ B.T -> (batch, r)
            lora_inter = x_d @ self.B.t()  # (batch, r)
            lora_out = lora_inter @ self.A.t()  # (batch, out)
            return base + self.scaling * lora_out
        else:
            # either r==0 or merged, just return base
            return base

    def merge(self):
        """Merge LoRA weights into the base linear weight (in-place).
           After merge, we set merged=True so forward uses only base weight.
        """
        if self.r <= 0 or self.merged:
            return
        # delta_W = A @ B  -> shape (out, in)
        delta = (self.A @ self.B) * self.scaling  # (out, in)
        # Add to base weight (in-place)
        self.linear.weight.data += delta
        self.merged = True

    def unmerge(self):
        """Revert the merge (subtract delta from base weight)."""
        if self.r <= 0 or not self.merged:
            return
        delta = (self.A @ self.B) * self.scaling
        self.linear.weight.data -= delta
        self.merged = False

def _should_replace(name: str, target_modules: Optional[List[str]]):
    if not target_modules:
        return False
    for t in target_modules:
        if t in name:
            return True
    return False

def apply_lora(model: nn.Module, config: LoraConfig):
    """
    Replace target Linear modules in `model` with LoRALinear wrappers.
    `target_modules` matches substring in module name.
    """
    # 1) collect candidate names first (avoid in-place iteration issues)
    candidates = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _should_replace(name, config.target_modules):
            candidates.append((name, module))

    # 2) replace them by setting attribute on parent
    for name, module in candidates:
        # find parent module
        parts = name.split('.')
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        last = parts[-1]
        orig_linear = getattr(parent, last)
        # create LoRA wrapper using same weight/bias
        lora_layer = LoRALinear(orig_linear, r=config.r, alpha=config.lora_alpha, dropout=config.lora_dropout)
        # set attribute
        setattr(parent, last, lora_layer)

        # If inference_mode and user expects merged weights now, optionally merge
        if config.inference_mode:
            lora_layer.merge()

    return model

def set_lora_trainable(model: nn.Module):
    """
    Freeze base model params, unfreeze LoRA params (A,B).
    Call after apply_lora.
    """
    for n, p in model.named_parameters():
        p.requires_grad = False
    # unfreeze A, B in LoRALinear
    for m in model.modules():
        if isinstance(m, LoRALinear):
            if m.A is not None:
                m.A.requires_grad = True
            if m.B is not None:
                m.B.requires_grad = True
            # optionally you may want bias trainable; left as False here.

def merge_all_lora(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()

def unmerge_all_lora(model: nn.Module):
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()

# Utilities to save/load only LoRA params
def lora_state_dict(model: nn.Module):
    """Return state_dict containing only LoRA params (A and B) keyed by module path."""
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + ('.' if name else '')
            if module.A is not None:
                sd[prefix + 'lora_A'] = module.A.detach().cpu()
            if module.B is not None:
                sd[prefix + 'lora_B'] = module.B.detach().cpu()
    return sd

def load_lora_state_dict(model: nn.Module, lora_sd: dict, strict: bool=True):
    """Load LoRA params from lora_sd produced by lora_state_dict."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + ('.' if name else '')
            a_key = prefix + 'lora_A'
            b_key = prefix + 'lora_B'
            if a_key in lora_sd:
                module.A.data.copy_(lora_sd[a_key].to(module.A.device))
            elif strict and module.A is not None:
                raise KeyError(f"Missing {a_key} in LoRA state dict")
            if b_key in lora_sd:
                module.B.data.copy_(lora_sd[b_key].to(module.B.device))
            elif strict and module.B is not None:
                raise KeyError(f"Missing {b_key} in LoRA state dict")

```

#### 3.2.1 LoRAConfig
```python
@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: List[str] = None  # list of substrings to match module names
    inference_mode: bool = False
```
这一部分确定了LoRA的各种参数

#### 3.2.2 apply_lora
该函数的功能是：  
遍历整个模型，找到所有需要替换的 nn.Linear 层。  
用 LoRALinear 层（LoRA 适配器）替换原有的 Linear 层。  
如果配置中指定了 inference_mode，则进行权重合并，优化推理时的计算。

详细解析
1) 收集待替换的候选模块
```python
candidates = []
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and _should_replace(name, config.target_modules):
        candidates.append((name, module))
```

通过 model.named_modules() 遍历模型的所有子模块。named_modules() 会返回模型中每个子模块的名称 name 和模块对象 module。  
isinstance(module, nn.Linear)：过滤出所有 nn.Linear 层，表示只关注线性层（全连接层）。  
_should_replace(name, config.target_modules) 是一个函数（未提供实现），用于判断模块名称是否符合配置中指定的目标模块 target_modules（即名字中包含特定子串的模块），如果符合则把这个模块记录为候选。

2) 替换为 LoRALinear 层
```python
for name, module in candidates:
    # 找到模块的父模块
    parts = name.split('.')
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    last = parts[-1]
    orig_linear = getattr(parent, last)
```

对每个候选的 Linear 层（name 和 module），首先通过 name.split('.') 把模块名拆分成各级层次的名称（例如 encoder.layer.0.attn.self.query）。

然后，从模型的顶层开始逐级向下获取父模块。getattr(parent, p) 获取 parent 模块中名为 p 的属性（模块）。这一过程会一直深入，直到找到目标模块的父模块。

last = parts[-1] 是模块的最终名称（即模块名），例如 weight、bias 等。

##### 创建 LoRA 包装器

LoRALinear 是用来包装原始 Linear 层的 LoRA 层，它会使用原始 Linear 层的权重和偏置，但会在其基础上引入低秩适配（LoRA）。

参数：
```python
r=config.r：低秩矩阵的秩，即 LoRA 的维度。

alpha=config.lora_alpha：LoRA 的缩放因子，影响训练过程中的调整强度。

dropout=config.lora_dropout：用于 LoRA 适配的 dropout。
```

3) 替换模块
##### 设置父模块的属性为 LoRA 包装层  
setattr(parent, last, lora_layer)：将 lora_layer 设置为父模块的一个属性，这相当于用 LoRA 层替换了原来的 Linear 层。

4) 如果是推理模式，合并 LoRA 权重
```python
if config.inference_mode:
    lora_layer.merge()
```

如果配置中的 inference_mode=True，意味着用户在推理过程中不希望继续使用 LoRA 层的额外计算，而是希望将 LoRA 适配的权重合并到原始的 Linear 权重中。  
lora_layer.merge()：合并 LoRA 层的适配权重到原始权重，避免推理时的额外计算开销。  
最后，返回修改后的模型：  
return model
 
这段代码的核心任务是：  
遍历并找到符合条件的 nn.Linear 层。  
将它们替换为 LoRALinear 层，该层可以进行低秩适配。  
如果是推理模式，则自动合并 LoRA 层的权重，避免推理时计算开销。  

