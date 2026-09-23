# 分组查询注意力（Grouped-Query Attention, GQA）

这份 bonus 资料展示了相比普通多头注意力（Multi-Head Attention, MHA），使用分组查询注意力（Grouped-Query Attention, GQA）所能节省的内存。

&nbsp;
## 引言（Introduction）

近年来，分组查询注意力（GQA）已经逐渐成为多头注意力（MHA）的一种新的、更节省计算和参数的标准替代方案。其实它并不是新东西，可以追溯到 2023 年的论文 [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)。就连老牌的 Llama 2 系列中的较大变体也已经在用它。

下面给一个简短的 GQA 总结：与 MHA（每个 head 都有一套自己的 key 和 value）不同，GQA 把多个 head 分组，让它们共享同一组 key 和 value projection，从而降低内存占用。

如下图所示，如果有 3 个 key-value 组、6 个 attention head，那么 head 1 和 head 2 共享一组 key 和 value，head 3 和 head 4、head 5 和 head 6 各自共享另一组 key 和 value。

&nbsp;

![GQA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/1.webp?1)

&nbsp;

这种 key 和 value 的共享减少了 key 和 value 计算的总次数，从而降低内存占用并提高效率。

总结一下，GQA 的核心思想是：通过让多个 query head 共享，减少 key 和 value head 的数量。这样做（1）降低了模型的参数总量，（2）在推理期间降低了 key 和 value tensor 的内存带宽——因为需要存进 KV cache、从 KV cache 取出的 key 和 value 变少了。

GQA 主要是一种针对 MHA 的计算效率优化，但消融实验（例如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)）表明，在 LLM 建模性能方面，它与标准 MHA 表现相当。

不过这要求 key-value 组数选得合理。极端情况下，如果所有 attention head 共用同一个 key-value 组，这就是所谓的多查询注意力（multi-query attention），内存占用会下降得更厉害，但建模性能可能会下降。（再极端一些，如果把 key-value 组数设成等于 query head 数，那就退化回了标准多头注意力。）

&nbsp;
## GQA 节省的内存（GQA Memory Savings）

节省的内存主要反映在 KV 存储上。我们可以用下面这个公式来计算 KV 存储大小：

bytes ≈ batch_size × seqlen × (embed_dim / n_heads) × n_layers × 2 (K,V) × bytes_per_elem × n_kv_heads

你可以用本目录下的 [memory_estimator_gqa.py](memory_estimator_gqa.py) 脚本，把这个公式套到不同的模型配置上，看 GQA 相比 MHA 能省多少内存：

```bash
➜ uv run memory_estimator_gqa.py \
  --emb_dim 4096 --n_heads 32 --n_layers 32 \
  --context_length 32768 --n_kv_groups 4 \
  --batch_size 1 --dtype bf16
==== Config ====
context_length   : 32768
emb_dim          : 4096
n_heads          : 32
n_layers         : 32
n_kv_groups      : 4
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 128
GQA n_kv_heads   : 8

==== KV-cache totals across all layers ====
MHA total KV cache  : 17.18 GB
GQA total KV cache  : 4.29 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
```

下图进一步展示了在不同 key-value 组数下，GQA 相对 MHA 节省的内存随上下文长度的变化：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gqa-memory/3.webp?4" alt="GQA" width="500px" />

&nbsp;

可以用 `uv run plot_memory_estimates_gqa.py` 复现这张图。

&nbsp;
## GQA 代码示例（GQA Code Examples）

本目录中的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_gqa.py](gpt_with_kv_gqa.py) 给出了 hands-on 示例：在 GPT 模型实现下对比 MHA 和 GQA 的内存占用。

注意，GQA 也用在 [Llama 3](../../ch05/07_gpt_to_llama)、[Gemma 3](../../ch05/12_gemma3) 和 [Qwen3](../../ch05/11_qwen3) 的 bonus 资料里。为了简洁起见，本目录里的代码脚本修改的是 GPT 架构——传统上 GPT 是不用 GQA 的。

注意这里的模型没有训练，所以生成的是无意义的文本。但你可以把它当作第 5-7 章里标准 GPT 模型的替代品直接训练。

另外，这个实现用到了 [另一节 bonus](../03_kv-cache) 中讲解的 KV cache，所以内存节省效果更明显。

```bash
uv run gpt_with_kv_mha.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12

...

Time: 453.81 sec
72 tokens/sec
Max memory allocated: 1.54 GB
```

```bash
uv run gpt_with_kv_gqa.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--n_kv_groups 4

...

Time: 516.33 sec
63 tokens/sec
Max memory allocated: 0.63 GB
```

为什么我们看到的节省幅度没有上面那张图里那么大，原因有两点：

1. 我用了一个较小的配置，让生成能在合理的时间内跑完。
2. 更重要的是，我们这里看到的是整个模型的内存，不只是注意力机制——模型里的全连接层占据了大部分内存（不过这是单独另一个话题了）。