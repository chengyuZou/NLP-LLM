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

