# 多头潜在注意力（Multi-Head Latent Attention, MLA）

这份 bonus 资料展示了相比普通多头注意力（MHA），使用多头潜在注意力（Multi-Head Latent Attention, MLA）所能节省的内存。

&nbsp;
## 引言（Introduction）

在 [../04_gqa](../04_gqa) 里，我们把分组查询注意力（GQA）当作 MHA 的一种计算效率优化方案。消融实验（例如 [原始 GQA 论文](https://arxiv.org/abs/2305.13245) 和 [Llama 2 论文](https://arxiv.org/abs/2307.09288)）表明，在 LLM 建模性能方面，GQA 与标准 MHA 表现相当。

接下来要讲的多头潜在注意力（MLA），被 [DeepSeek V2、V3 和 R1](https://arxiv.org/abs/2412.19437) 所使用，提供了一种不同的内存节省策略，并且特别适合与 KV cache 配合。MLA 没有像 GQA 那样共享 key 和 value head，而是在存入 KV cache 之前，把 key 和 value tensor 压缩到一个更低维的空间。

推理时，这些压缩后的 tensor 会被投影回原始大小再使用，如下图所示。虽然多了一次矩阵乘法，但内存占用降低了。

&nbsp;

![MLA](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/1.webp)

&nbsp;

（顺便提一句，query 在训练时也会被压缩，但推理时不压缩。）

前面也提过，MLA 在 DeepSeek V3 中并不是首次出现——它的前身 [DeepSeek V2](https://arxiv.org/abs/2405.04434) 就已经用上了（甚至可以说是在 V2 中首次提出）。V2 论文里还有一些有意思的消融实验，可能能解释为什么 DeepSeek 团队选了 MLA 而不是 GQA（见下图）。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/2.webp" alt="GQA" width="500px" />

&nbsp;

如上图所示，GQA 的表现看起来比 MHA 差，而 MLA 在建模性能上反而略优于 MHA——这很可能就是 DeepSeek 团队选择 MLA 而不是 GQA 的原因。（如果能看到 MLA 和 GQA 之间 "每 token 的 KV cache" 节省对比就更好了！）

总结一下这一节内容，MLA 是一个聪明的做法：在略微超越 MHA 建模性能的前提下，显著降低 KV cache 内存。

&nbsp;
## MLA 节省的内存（MLA Memory Savings）

节省的内存主要反映在 KV 存储上。我们可以用下面这个公式来计算：

bytes ≈ batch_size × seqlen × n_layers × latent_dim × bytes_per_elem

对比一下，MHA 的 KV cache 内存是这么算的：

bytes ≈ batch_size × seqlen × n_layers × embed_dim × 2 (K,V) × bytes_per_elem

也就是说，在 MLA 里，我们把 "embed_dim × 2 (K,V)" 压缩成了 "latent_dim"，因为如前面图示，我们只存压缩后的 latent 表示，而不是完整的 key 和 value 向量。


可以用本目录下的 [memory_estimator_mla.py](memory_estimator_mla.py) 脚本，把这个公式套到不同的模型配置上，看 MLA 相对 MHA 能省多少内存：

```bash
➜ uv run memory_estimator_mla.py \
  --context_length 8192 \
  --emb_dim 2048 \
  --n_heads 24 \
  --n_layers 48 \
  --n_kv_groups 4 \
  --batch_size 1 \
  --dtype bf16 \
  --latent_dim 1024
==== Config ====
context_length   : 8192
emb_dim          : 2048
n_heads          : 24
n_layers         : 48
n_kv_groups      : 4
latent_dim       : 1024
batch_size       : 1
dtype            : bf16 (2 Bytes/elem)
head_dim         : 86
GQA n_kv_heads   : 6

==== KV-cache totals across all layers ====
MHA total KV cache  : 3.25 GB
GQA total KV cache  : 0.81 GB
MLA total KV cache  : 0.81 GB
Ratio (MHA / GQA)   : 4.00x
Savings (GQA vs MHA): 75.00%
Ratio (MHA / MLA)   : 4.03x
Savings (MLA vs MHA): 75.19%
```

注意上面的压缩比（`--emb_dim 2048 -> latent_dim 1024`），效果和 GQA 类似。实际上，压缩比是一个需要仔细调的超参数——`latent_dim` 选得过小会对建模性能产生负面影响（类似在 GQA 里 `n_kv_groups` 选得过多的情况）。

下图进一步展示了在不同 `latent_dim` 下，MLA 相对 MHA 节省的内存随上下文长度的变化：

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mla-memory/3.webp?2" alt="GQA" width="500px" />

&nbsp;

可以用 `uv run plot_memory_estimates_mla.py` 复现这张图。


&nbsp;
## MLA 代码示例（MLA Code Examples）

本目录下的 [gpt_with_kv_mha.py](gpt_with_kv_mha.py) 和 [gpt_with_kv_mla.py](gpt_with_kv_mla.py) 给出了 hands-on 示例：在 GPT 模型实现下对比 MHA 和 MLA 的内存占用。

这里的 MLA 代码参考了 [https://huggingface.co/bird-of-paradise/deepseek-mla](https://huggingface.co/bird-of-paradise/deepseek-mla) 的实现。

注意 MLA 也可以和 [GQA](../04_gqa) 组合使用，但为了简洁，这里没有这样做。（截至目前，我也没看到主流 LLM 这样用。）

另外注意模型没有训练，所以生成的是无意义文本。不过可以用作第 5-7 章里标准 GPT 模型的替代品直接训练。

最后，这个实现用到了 [另一节 bonus](../03_kv-cache) 中讲解的 KV cache，所以内存节省效果更明显。

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
uv run gpt_with_kv_mla.py \
--max_new_tokens 32768 \
--n_heads 24 \
--n_layers 12 \
--emb_dim 768 \
--latent_dim 192 # (768×2)/192 = 8× compression

...

Time: 487.21 sec
67 tokens/sec
Max memory allocated: 0.68 GB
```

为什么我们看到的节省幅度没有上面那张图里那么大，原因有两点：

1. 我用了一个较小的配置，让生成能在合理的时间内跑完。
2. 更重要的是，我们这里看到的是整个模型的内存，不只是注意力机制——模型里的全连接层占据了大部分内存（不过这是单独另一个话题了）。