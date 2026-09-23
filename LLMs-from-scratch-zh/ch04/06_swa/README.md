# 滑动窗口注意力（Sliding Window Attention, SWA）

这份 bonus 资料展示了相比普通多头注意力（MHA），使用滑动窗口注意力（Sliding Window Attention, SWA）所能节省的内存。


&nbsp;
## 引言（Introduction）

什么是滑动窗口注意力（SWA）？如果把普通 self-attention 看作一种 *全局* 注意力机制——因为每个序列元素都能访问其它所有序列元素——那么 SWA 可以看成一种 *局部* 注意力，因为它把当前 query 位置周围的上下文大小做了限制，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/1.webp?2" alt="Sliding Window Attention" width="500px" />

如上图所示，每个 token 不再关注所有前文 token，而只关注当前位置附近一个固定大小的局部窗口。这种局部化的注意力机制能让 KV cache 的规模显著缩小。

在剩下的引言部分，我们将以 [Gemma 3](https://arxiv.org/abs/2503.19786) 为背景讨论 SWA，[../../ch05/12_gemma3](../../ch05/12_gemma3) 中有它的从零实现。

滑动窗口注意力最早出现在 [2020 年的 LongFormer 论文](https://arxiv.org/abs/2004.05150)。我们这里重点关注 Google 的 Gemma 模型，是因为它们是很好的开放权重模型，证明 SWA 在近期大模型中确实可行。

[Gemma 2](https://arxiv.org/abs/2408.00118) 使用了局部（SWA）和全局 attention 层 1:1 混合的方式。每个 token 可以关注 4k token 的上下文窗口。这种 1:1 混合是出于效率和全局上下文建模能力的平衡——纯局部 attention 对 LLM 来说可能过于受限。

[Gemma 3](https://arxiv.org/abs/2503.19786) 则把这一设计进一步推向效率。它使用了 5:1 的 SWA 与全局 attention 层比例，也就是说每 5 层 SWA 配 1 层全局 attention。同时，滑动窗口大小从 Gemma 2 的 4096 token 缩小到了 1024 token。

有趣的是，Gemma 3 技术报告里的消融实验表明，这些改动对整体模型质量只有轻微影响。换言之，SWA 带来的巨大内存和计算节省几乎不损失建模性能。


&nbsp;
## 滑动窗口注意力（SWA）的内存节省（Sliding Window Attention (SWA) Memory Savings）

节省的内存主要反映在 KV 存储上。我们可以用下面这个公式来计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

使用 SWA 时，我们把上面的 seqlen 替换成窗口大小 W。也就是说，使用 SWA 时，KV cache 的规模缩小到原来的 "W / seqlen"。（为简洁起见，这里假设每一层都用了 SWA。）


可以用本目录下的 [memory_estimator_swa.py](memory_estimator_swa.py) 脚本，把这个公式套到不同的模型配置上，看 SWA 相对 MHA 能省多少内存：

```bash
➜ uv run memory_estimator_swa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 1024 --swa_ratio "5:1"
==== Config ====
context_length         : 32768
sliding_window_size    : 1024
emb_dim                : 4096
n_heads                : 32
n_layers               : 32
n_kv_groups            : 4
batch_size             : 1
dtype                  : bf16 (2 Bytes/elem)
head_dim               : 128
GQA n_kv_heads         : 8
Effective SWA window W : 1024
Layer ratio (SWA:Full) : 5:1
Distributed layers     : 27 SWA, 5 FULL

==== KV-cache totals across all layers ====
MHA KV total           : 17.18 GB
GQA KV total           : 4.29 GB
MHA + SWA (Ratio: 5:1) : 3.14 GB
GQA + SWA (Ratio: 5:1) : 0.78 GB
```

注意 Gemma 3 把 SWA 和 GQA 组合在一起用。

下图进一步展示了在不同上下文长度下，SWA 相对 MHA 节省的内存：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/swa-memory/4.webp?2" alt="SWA" width="800px" />

&nbsp;

可以用如下命令复现这些图：

```bash
uv run plot_memory_estimates_swa.py \
  --emb_dim 4096 --n_heads 48 --n_layers 36 \
  --batch_size 1 --dtype bf16 \
  --sliding_window_size 2048 --swa_ratio "5:1"
```


&nbsp;
## SWA 代码示例（SWA Code Examples）

本目录下的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_swa.py](gpt_with_kv_swa.py) 给出了 hands-on 示例：在 GPT 模型实现下对比 MHA 和 SWA 的内存占用。

注意 SWA 也可以和 MLA、GQA 组合使用（如前所述），但为了简洁，这里没有这样做。

注意模型没有训练，所以生成的是无意义文本。但你可以把它当作第 5-7 章里标准 GPT 模型的替代品直接训练。

另外，这个实现用到了 [另一节 bonus](../03_kv-cache) 中讲解的 KV cache，所以内存节省效果更明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_swa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--sliding_window_size 1024 \
--sliding_window_stride 5   # 像 Gemma 3

...

Time: 514.38 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

为什么我们看到的节省幅度没有上面那张图里那么大，原因有两点：

1. 我用了一个较小的配置，让生成能在合理的时间内跑完。
2. 更重要的是，我们这里看到的是整个模型的内存，不只是注意力机制——模型里的全连接层占据了大部分内存（不过这是单独另一个话题了）。