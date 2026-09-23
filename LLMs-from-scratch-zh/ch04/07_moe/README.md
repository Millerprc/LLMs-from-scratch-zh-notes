# 专家混合（Mixture of Experts, MoE）

这份 bonus 资料展示了相比普通 feed-forward（FFN）层，使用专家混合（Mixture-of-Experts, MoE）层所能节省的（每 token 的）内存。



&nbsp;
## 引言（Introduction）

MoE 的核心思想是：把 transformer block 中的每一个 feed-forward 模块替换成多个 expert 层，每个 expert 也是一个 feed-forward 模块。也就是说，我们把单个 feed-forward 块换成多个 feed-forward 块，如下图所示。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/1.webp" alt="SWA" width="800px" />

transformer block 中的 feed-forward 块（上图中的深灰色块）通常包含模型总参数的很大一部分。（注意 transformer block 在 LLM 中会重复很多次；以 DeepSeek-V3 为例，重复了 61 次。）

所以，把 *单个* feed-forward 块换成 *多个* feed-forward 块（就像 MoE 那样）会显著增加模型的总参数量。但关键的技巧是：对每个 token，我们并不使用（激活）所有 expert——一个路由器（router）只为每个 token 选中少数几个 expert。

因为一次只有少数几个 expert 处于激活状态，MoE 模块通常被称为 *稀疏*（sparse）模块，对应于 *稠密*（dense）模块——后者每次都用全部参数。但 MoE 让 LLM 拥有更大的总参数容量，也就是说它能在训练时吸收更多知识。稀疏性让推理保持高效，因为我们不会同时使用所有参数。

比如 DeepSeek-V3 每个 MoE 模块有 256 个 expert，总参数量 6710 亿。但推理时每次只有 9 个 expert 处于激活状态（1 个 shared expert 加 router 选出的 8 个）。也就是说每个 token 推理步只用 370 亿参数，而不是全部 6710 亿。

DeepSeek-V3 的 MoE 设计中有一个值得注意的特性：shared expert。这是每个 token 永远激活的一个 expert。这个想法并不新鲜，早在 [2022 年的 DeepSpeed-MoE](https://arxiv.org/abs/2201.05596) 和 [2024 年的 DeepSeek MoE](https://arxiv.org/abs/2401.06066) 论文中就已经出现过。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/3.webp?1" alt="MoE shared expert" width="500px" />

（这张标注图来自 [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) 论文。）

&nbsp;

shared expert 带来的好处首先在 [DeepSpeed-MoE 论文](https://arxiv.org/abs/2201.05596) 中被注意到——文中发现，相比没有 shared expert，它能提升整体建模性能。这大概是因为常见或重复的模式不再需要由多个独立 expert 各自学习，这样它们就有更多容量去学习更专门化的模式。

&nbsp;
## MoE 节省的内存（Mixture of Experts (MoE) Memory Savings）

MoE 模型节省的内存主要来自减少的激活存储和计算。在普通的（稠密）FFN 中，每个 token 都要用完整的中间维度。

而 MoE 层只为每个 token 路由到一小部分 expert（例如 `top_k` 个 `num_experts`）。

使用 MoE 层时，每个 token 只有 `top_k` 个 expert 处于激活，所以相对于同总容量的稠密 FFN，有效内存（和计算）大致按 `top_k / num_experts` 的比例缩小。

可以用本目录下的 [memory_estimator_moe.py](memory_estimator_moe.py) 脚本，把这个公式套到不同的模型配置上，看 MoE 相对 FFN 能省多少内存（注意这里算的是单个 transformer block；要算模型整体的节省，乘上 transformer block 的数量）：

```bash
uv run memory_estimator_moe.py --emb_dim 7168 --hidden_dim 14336 --ffn_type swiglu \
  --num_experts 8 --top_k 2 --match_dense 
==== Config ====
emb_dim                : 7168
hidden_size            : 14336
ffn_type               : swiglu
num_experts            : 8
top_k                  : 2
dtype                  : bf16 (2 Bytes/elem)
match_dense            : True

==== Model weights (parameters) ====
Dense FFN params       : 308,281,344 (0.62 GB)
Per-expert params      : 38,535,168 (0.08 GB)
Router params          : 57,344 (0.00 GB)
MoE TOTAL params       : 308,338,688 (0.62 GB)
MoE ACTIVE/Token       : 77,127,680 (0.15 GB)
moe_hidden_size        : 1792
```

从上面的结果可以看到，对于一个输入/输出维度（`emb_dim`）为 7168、中间层维度（`hidden_dim`）为 14336 的 FFN，单层有约 3.08 亿参数，并且这些参数在前向时都是激活的。

现在，如果我们用一个参数量大致相当（约 3.08 亿）的 MoE 层，配 8 个 expert、激活 2 个 expert，那么每次前向时只有约 7700 万 固定参数处于激活。

更重要的是，在 expert 总数不变的情况下，expert 越多，激活参数越少，"节省"越大：

&nbsp;

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/moe-memory/2.webp" alt="SWA" width="500px" />



&nbsp;

可以用如下命令复现这张图：

```bash
uv run plot_memory_estimates_moe.py \
    --emb_dim 7168 \
    --hidden_dim 28672 \
    --ffn_type swiglu \
    --top_k 8
```


&nbsp;
## MoE 代码示例（MoE Code Examples）

本目录下的 [gpt_with_kv_ffn.py](gpt_with_kv_ffn.py) 和 [gpt_with_kv_moe.py](gpt_with_kv_moe.py) 给出了 hands-on 示例：在 GPT 模型实现下对比普通 FFN 和 MoE 的内存占用。注意这两份脚本都使用了 [SwiGLU](https://arxiv.org/abs/2002.05202) feed-forward 模块（见本页面第一张图；GPT-2 传统上用 GELU）。

**注意：模型没有训练，所以生成的是无意义文本。在 [../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb](../../ch05/11_qwen3/standalone-qwen3-moe-plus-kvcache.ipynb) 的 bonus 资料中可以找到一个训练过的 MoE。**



首先，我们用普通 FFN 跑一遍模型：


```bash
uv run gpt_with_kv_ffn.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 32768

...
Avg FFN time/call: 0.759 ms
Avg FFN mem delta/call: 0.19 MB (max 0.75 MB)
...
Time: 25.13 sec
40 tokens/sec
Max memory allocated: 11.47 GB
```

为了和 MoE 公平比较，我们需要把 expert 的尺寸缩小。比如用 32 个 expert 时，要把 `--hidden_dim` 设为 `32768/32`：

```bash
uv run gpt_with_kv_moe.py \
--max_new_tokens 1024 \
--n_heads 16 \
--n_layers 12 \
--emb_dim 4096 \
--hidden_dim 1024 \
--num_experts 32 \
--num_experts_per_tok 2

...
Avg MoE FF time/call: 1.555 ms
Avg MoE FF mem delta/call: 0.04 MB (max 0.11 MB)
...
Time: 35.11 sec
29 tokens/sec
Max memory allocated: 11.48 GB
```

可以看到，稠密 feed-forward 层处理一个 token 大约要 0.76 ms，激活占用大约 0.19 MB（峰值约 0.75 MB）。

稀疏的 MoE 层只保留约 0.04 MB 内存（峰值 0.11 MB）。不过这是以大约两倍的前馈计算时间为代价的。（多出来的路由开销是一个原因，我的实现可能也不是最高效的。）

两种情况下整体生成的峰值 GPU 内存都在 11.5 GB 左右，因为两种版本加载的权重参数数量相同、KV cache 大小相同，这两者在这里是主导项。

总之，这里能清楚看到 MoE 的权衡：用大约 2 倍的 feed-forward 计算时间，换取约 4-5 倍的 FFN 内存节省。

注意，如果我们一次处理更多 token（比如 batch size > 1，这里为了代码简洁没用 batch），节省效果会更明显。