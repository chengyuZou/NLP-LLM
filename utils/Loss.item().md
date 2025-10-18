## .item()方法是，取一个元素张量里面的具体元素值并返回该值，可以将一个零维张量转换成int型或者float型，在计算loss，accuracy时常用到。

### 作用:
#### 1.item（）取出张量具体位置的元素元素值
#### 2.并且返回的是该位置元素值的高精度值
#### 3.保持原元素类型不变；必须指定位置
#### 4.节省内存(不会计入计算图)

```python
import torch
 
loss = torch.randn(2, 2)
 
print(loss)
print(loss[1,1])
print(loss[1,1].item())
```
```python
输出结果
tensor([[-2.0274, -1.5974],
        [-1.4775,  1.9320]])
tensor(1.9320)
1.9319512844085693
```

## 注意事项  
**防止显存爆炸**：在训练过程中，如果直接将损失值累加（例如 `loss_sum += loss`），由于 PyTorch 的动态图机制，这会导致显存不断增加，因为累加的损失值会被视为计算图的一部分。  
为了避免这个问题，可以使用 `loss.item()` 来获取损失值的标量，然后进行累加，这样可以防止显存的无限增长。

## 参考文献
### **[pytorch学习：loss为什么要加item()](https://blog.csdn.net/github_38148039/article/details/107144632?spm=1001.2101.3001.6661.1&utm_medium=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-107144632-blog-104333552.pc_relevant_multi_platform_whitelistv4&depth_1-utm_source=distribute.pc_relevant_t0.none-task-blog-2~default~CTRLIST~Rate-1-107144632-blog-104333552.pc_relevant_multi_platform_whitelistv4&utm_relevant_index=1)**
### **[loss.item()用法和注意事项详解](https://blog.csdn.net/Viviane_2022/article/details/128379670?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522f6dd4a09d984f15e21918f348c6a588f%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=f6dd4a09d984f15e21918f348c6a588f&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~top_positive~default-1-128379670-null-null.142^v102^pc_search_result_base1&utm_term=loss.item%28%29&spm=1018.2226.3001.4187)**
### **[pytorch 和tensorflow loss.item()` 只能用于只有一个元素的张量. 防止显存爆炸](https://blog.csdn.net/zhangfeng1133/article/details/144003084?ops_request_misc=&request_id=&biz_id=102&utm_term=loss.item()&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-144003084.142^v102^pc_search_result_base1&spm=1018.2226.3001.4187)**

