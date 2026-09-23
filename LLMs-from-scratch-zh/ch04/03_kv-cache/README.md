# Bonus Material: 键值缓存（KV Cache）



**本目录实现了为 GPT 模型添加键值缓存（KV cache）的代码。**

&nbsp;
## 概览（Overview）

简而言之，KV cache 把推理时已经计算过的中间 key（K）和 value（V）保存下来以便复用，能在文本生成时带来显著加速。代价是会让代码复杂一些、增加内存占用，并且不能在训练阶段使用。但在部署 LLM 时，推理速度的提升通常足以抵消代码复杂度和内存带来的成本。

&nbsp;
## 它是如何工作的（How it works）

假设 LLM 正在生成文本。比如给它下面的 prompt："Time flies"。

下面这张图展示了底层注意力分数计算的一个摘录（沿用第 3 章的图示，标出了 key 和 value 向量）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-1.png?3" width=800>

正如第 2 章和第 4 章所学，LLM 一次生成一个词（或一个 token）。假设 LLM 生成了单词 "fast"，那么下一轮的 prompt 就变成了 "Time flies fast"。下图展示了这一情形：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/kv-cache-attn-2.png?3" width=800>

对比前面两张图可以看到，前两个 token 的 key 和 value 向量是完全相同的。如果在每一轮生成时都重新计算它们，就是一种浪费。

所以 KV cache 的核心思想就是实现一个缓存机制：把之前生成过的 key 和 value 向量存起来复用，避免不必要的重复计算。

&nbsp;

## KV cache 的实现（KV cache implementation）

实现 KV cache 的方法很多，核心思想都是：每一轮生成时，只为新生成的 token 计算 key 和 value tensor。

我选择了一种简单、便于阅读代码的实现。我觉得最直观的方式就是直接浏览代码改动，看它具体是怎么实现的。

本目录里有两份代码文件：

1. [`gpt_ch04.py`](gpt_ch04.py)：取自第 3 章和第 4 章的自包含代码，实现 LLM 并运行简单的文本生成函数
2. [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)：在上面的基础上做了必要的修改，加入 KV cache

你可以任选其一：

a. 打开 [`gpt_with_kv_cache.py`](gpt_with_kv_cache.py)，找到标记为 `# NEW` 的部分，它们就是新增改动：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/new-sections.png?3" width=800>

b. 或者用你喜欢的 diff 工具对比这两份代码：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/kv-cache/file-diff.png?3" width=800>

下面是一个简短的实现要点回顾。

&nbsp;

### 1. 注册 cache buffer

在 `MultiHeadAttention` 构造函数里新增两个 buffer：`cache_k` 和 `cache_v`，用来在多个生成步骤间累积 key 和 value：

```python
self.register_buffer("cache_k", None)
self.register_buffer("cache_v", None)
```

&nbsp;

### 2. 带有 `use_cache` 标志的 forward

接着扩展 `MultiHeadAttention` 的 `forward` 方法，让它接收 `use_cache` 参数。把新的一段 token 投影成 `keys_new`、`values_new` 和 `queries` 之后，要么初始化 KV cache，要么向已有 cache 中追加：

```python
def forward(self, x, use_cache=False):
    b, num_tokens, d_in = x.shape

    keys_new = self.W_key(x)  # Shape: (b, num_tokens, d_out)
    values_new = self.W_value(x)
    queries = self.W_query(x)
    #...

    if use_cache:
        if self.cache_k is None:
            self.cache_k, self.cache_v = keys_new, values_new
        else:
            self.cache_k = torch.cat([self.cache_k, keys_new], dim=1)
            self.cache_v = torch.cat([self.cache_v, values_new], dim=1)
        keys, values = self.cache_k, self.cache_v
    else:
        keys, values = keys_new, values_new
        
    # ...
    
    num_tokens_Q = queries.shape[-2]
    num_tokens_K = keys.shape[-2]
    if use_cache:
        mask_bool = self.mask.bool()[
            self.ptr_current_pos:self.ptr_current_pos + num_tokens_Q, :num_tokens_K
        ]
        self.ptr_current_pos += num_tokens_Q
    else:
        mask_bool = self.mask.bool()[:num_tokens_Q, :num_tokens_K]
```

&nbsp;


### 3. 清空 cache

文本生成时，在两段相互独立的序列之间（例如两段文本生成调用之间），必须重置两个 buffer。因此我们再给 `MultiHeadAttention` 加一个清空 cache 的方法：

```python
def reset_cache(self):
    self.cache_k, self.cache_v = None, None
    self.ptr_current_pos = 0
```

&nbsp;

### 4. 在整个模型里把 `use_cache` 一路传下去

对 `MultiHeadAttention` 改完之后，接下来还要修改 `GPTModel`。首先在构造函数里加上一个 token 位置追踪变量：

```python
self.current_pos = 0
```

然后，把原来一次调用每个 transformer block 的写法换成显式的循环，把 `use_cache` 透传到每个 block：

```python
def forward(self, in_idx, use_cache=False):
    # ...
 
    if use_cache:
        pos_ids = torch.arange(
            self.current_pos, self.current_pos + seq_len,            
            device=in_idx.device, dtype=torch.long
        )
        self.current_pos += seq_len
    else:
        pos_ids = torch.arange(
            0, seq_len, device=in_idx.device, dtype=torch.long
        )
    
    pos_embeds = self.pos_emb(pos_ids).unsqueeze(0)
    x = tok_embeds + pos_embeds
    # ...
    for blk in self.trf_blocks:
        x = blk(x, use_cache=use_cache)
```

上面的改动还要求 `TransformerBlock` 也做一点小修改，让它接收 `use_cache` 参数：
```python
    def forward(self, x, use_cache=False):
        # ...
        self.att(x, use_cache=use_cache)
```

最后，为了方便起见，给 `GPTModel` 加一个模型级的 reset，一次性清空所有 block 的 cache：

```python
def reset_kv_cache(self):
    for blk in self.trf_blocks:
        blk.att.reset_cache()
    self.current_pos = 0
```

&nbsp;

### 5. 在生成中使用 cache

在 `GPTModel`、`TransformerBlock`、`MultiHeadAttention` 都改完之后，下面的代码展示了在一个简单的文本生成函数中如何使用 KV cache：

```python
def generate_text_simple_cached(model, idx, max_new_tokens, 
                                context_size=None, use_cache=True):
    model.eval()
    ctx_len = context_size or model.pos_emb.num_embeddings

    with torch.no_grad():
        if use_cache:
            # 用完整 prompt 初始化 cache
            model.reset_kv_cache()
            logits = model(idx[:, -ctx_len:], use_cache=True)

            for _ in range(max_new_tokens):
                # a) 选取 log-prob 最高的 token（贪心采样）
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                # b) 把这个 token 拼到当前序列后面
                idx = torch.cat([idx, next_idx], dim=1)
                # c) 只把新 token 喂给模型
                logits = model(next_idx, use_cache=True)
        else:
            for _ in range(max_new_tokens):
                logits = model(idx[:, -ctx_len:], use_cache=False)
                next_idx = logits[:, -1].argmax(dim=-1, keepdim=True)
                idx = torch.cat([idx, next_idx], dim=1)

    return idx
```

注意第 c) 步，我们只把新 token `next_idx` 喂给模型：`logits = model(next_idx, use_cache=True)`。如果没有 cache，我们则要把全部输入 `logits = model(idx[:, -ctx_len:], use_cache=False)` 喂给模型，因为没有缓存可复用的 key 和 value。

&nbsp;

## 简单的性能对比（Simple performance comparison）

在概念层面讲完 KV cache 之后，关键问题是：在一个小例子上它实际效果如何？我们可以把上述两份代码文件当作普通 Python 脚本运行——会用那个 124M 参数的小 LLM 在给定 4-token prompt "Hello, I am" 的情况下生成 200 个新 token：

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt

python gpt_ch04.py

python gpt_with_kv_cache.py
```

在一台 M4 芯片的 Mac Mini（CPU）上，结果如下：

|                        | Tokens/sec |
| ---------------------- | ---------- |
| `gpt_ch04.py`          | 27         |
| `gpt_with_kv_cache.py` | 144        |

可以看到，在 124M 参数的小模型和仅 200 个 token 的短序列上，就已经获得了大约 5 倍的加速。（注意：这个实现优先考虑代码可读性，并未针对 CUDA 或 MPS 运行时做优化——如果要优化，通常会预先分配好 tensor，而不是每次重新创建并拼接它们。）

**注意：** 两种情况下模型输出的都是"胡言乱语"，看起来像这样：

> Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous bore ITVEGIN ministriesysics Kle functional recountrictionchangingVirgin embarrassedgl ...

因为模型还没训练。下一章会训练模型，届时你可以在训练好的模型上使用 KV cache（不过 KV cache 只在推理时使用）来生成连贯的文本。这里为了保持代码简洁，我们用未训练的模型来演示。

更重要的是：`gpt_ch04.py` 和 `gpt_with_kv_cache.py` 两个实现生成的文本是完全相同的。这说明 KV cache 实现是对的——一旦索引写错，结果就会不一致。

&nbsp;

## KV cache 的优缺点（KV cache advantages and disadvantages）

随着序列变长，KV cache 的收益和代价会按下面这些方式变得更明显：

- [Good] **计算效率提升**：没有 cache 时，第 *t* 步的注意力要把新 query 和前面 *t* 个 key 比对一遍，累计开销随序列长度按 O(n²) 增长。有了 cache，每个 key 和 value 只算一次然后复用，每一步的复杂度降到线性 O(n)。

- [Bad] **内存使用线性增长**：每个新 token 都要追加到 KV cache 里。序列越长、模型越大，累计的 KV cache 越大，可能占用相当大甚至无法承受的（GPU）内存。一种折中方案是截断 KV cache，但代价是再多一层复杂度（但部署 LLM 时，这种复杂度往往值得）。

&nbsp;
## 优化 KV cache 的实现（Optimizing the KV Cache Implementation）

上面这个强调优先实现重度的 KV cache 实现主要是为了清晰，优先考虑了代码可读性和教学目的。要把它部署到真实场景（尤其是更大的模型和更长的序列）中，还需要更仔细的优化。

&nbsp;
### 扩展 cache 时的常见陷阱（Common pitfalls when scaling the cache）

- **内存碎片和重复分配**：像前面那样不停地用 `torch.cat` 拼接 tensor，会因为反复申请和释放内存成为性能瓶颈。

- **内存使用线性增长**：如果处理不当，对于超长序列，KV cache 会大得不可用。

&nbsp;
#### 建议 1：预分配内存（Tip 1: Pre-allocate Memory）

与其反复拼接 tensor，不如根据预期的最大序列长度预先分配一块足够大的 tensor。这能让内存使用稳定并减少开销。用伪代码大致是这样的：

```python
# 示例：为 keys 和 values 预分配内存
max_seq_len = 1024  # 期望的最大序列长度
cache_k = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
cache_v = torch.zeros((batch_size, num_heads, max_seq_len, head_dim), device=device)
```

推理时只需要往这些预分配 tensor 的对应切片里写即可。

&nbsp;
#### 建议 2：用滑动窗口截断 cache（Tip 2: Truncate Cache via Sliding Window）

为了不让 GPU 内存爆掉，可以用滑动窗口加动态截断的方式，只保留最近 `window_size` 个 token：


```python
# 滑动窗口 cache 实现
window_size = 512
cache_k = cache_k[:, :, -window_size:, :]
cache_v = cache_v[:, :, -window_size:, :]
```

&nbsp;
#### 工程实践中的优化（Optimizations in practice）

这些优化都可以在 [`gpt_with_kv_cache_optimized.py`](gpt_with_kv_cache_optimized.py) 里看到。


在一台 M4 芯片的 Mac Mini（CPU）上，生成 200 个 token、窗口大小等于上下文长度（保证结果相同）的情况下，运行时间对比如下：

|                                  | Tokens/sec |
| -------------------------------- | ---------- |
| `gpt_ch04.py`                    | 27         |
| `gpt_with_kv_cache.py`           | 144        |
| `gpt_with_kv_cache_optimized.py` | 166        |

由于模型太小，CUDA 设备上的速度优势消失了——数据传输和通信的开销超过了 KV cache 在这种小模型上的收益。

&nbsp;
## 延伸阅读（Additional Resources）

1. [Qwen3 from-scratch KV cache 基准](../../ch05/11_qwen3#pro-tip-2-speed-up-inference-with-compilation)
2. [Llama 3 from-scratch KV cache 基准](../../ch05/07_gpt_to_llama/README.md#pro-tip-3-speed-up-inference-with-compilation)
3. [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) —— 本文 README 的更详细版本