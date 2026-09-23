# 用于线性注意力的 Gated DeltaNet（Gated DeltaNet for Linear Attention）

最近，[Qwen3-Next](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 和 [Kimi Linear](https://arxiv.org/abs/2510.26692) 提出了混合 transformer 架构，用注意力机制的替代方案——这些替代方案相对上下文长度是线性而不是二次复杂度。

Qwen3-Next 和 Kimi Linear 都使用 3:1 的比例，也就是说每三个采用 Gated DeltaNet 变体的 transformer block，配一个使用全注意力的 block，如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/01.webp" alt="Qwen3-Next versus Kimi Linear">



&nbsp;

## 引言与概览（Introduction and Overview）

Gated DeltaNet 是一种线性注意力变体，灵感来自循环神经网络（RNN），并包含一个门控机制——这个门控来自论文 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)。从某种意义上说，Gated DeltaNet 是带有 Mamba 风格门控的 DeltaNet，而 DeltaNet 本身就是一种线性注意力机制。

Kimi Linear 通过 Kimi Delta Attention（KDA）机制修改了 Qwen3-Next 的线性注意力机制，这本质上是对 Gated DeltaNet 的进一步改进。Qwen3-Next 用的是标量门控（每个 attention head 一个值）来控制记忆衰减率，而 Kimi Linear 把它换成了对每个特征维度的逐通道门控。作者们认为这给记忆控制提供了更细的粒度，从而能提升长上下文推理。

另外，对于全注意力层，Kimi Linear 把 Qwen3-Next 的 gated attention 层（基本就是带输出门控的标准多头注意力层）换成了多头潜在注意力（MLA）。这和我们之前在 DeepSeek V3/R1 一节中讨论过的 MLA 是同一个机制，只是多了一个门控。（回顾一下，MLA 压缩了 key/value 空间以减小 KV cache 大小。）

Kimi Linear 中的 MLA 没有使用门控——这是刻意的，这样作者能把这个架构和标准 MLA 更直接地进行对比；不过他们 [表示](https://x.com/yzhang_cs/status/1984631714464088563) 计划未来把它加上。

因为我们已经在 [../05_mla](../05_mla) 中实现了 MLA，这份 bonus 资料重点放在 Gated DeltaNet 上。


&nbsp;
## Gated Attention

在讲 Gated DeltaNet 本身之前，先简单说说门控。从上一张 Qwen3-Next 架构图的上半部分可以看到，Qwen3-Next 使用了"gated attention"。这其实就是普通 attention 层外加一个额外的 sigmoid 门控。

下面是我在第 3 章的 `MultiHeadAttention` 代码上做的简单修改，用来演示这个门控：

```python
import torch
from torch import nn

class GatedMultiHeadAttention(nn.Module):
    def __init__(
        self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Add gate
        self.W_gate = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
            persistent=False,
        )

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        ####################################################
        ### NEW: Add gate
        gate = self.W_gate(x)
        ####################################################
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)

        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(
            mask_bool, torch.finfo(attn_scores.dtype).min
        )

        attn_weights = torch.softmax(
            attn_scores / (self.head_dim ** 0.5), dim=-1
        )
        attn_weights = self.dropout(attn_weights)

        context = (attn_weights @ values).transpose(1, 2)
        context = context.reshape(b, num_tokens, self.d_out)

        ####################################################
        ### NEW: Add gate
        context = context * torch.sigmoid(gate)
        ####################################################
        out = self.out_proj(context)
        return out
```



可以看到，正常算完 attention 之后，模型用一个来自同一输入的独立门控信号、施加 sigmoid 把它限制在 0 到 1 之间，再与 attention 输出相乘。这样模型就能动态放大或缩小某些特征。Qwen3-Next 的开发者 [表示](https://qwen.ai/blog?id=4074cca80393150c248e508aa62983f9cb7d27cd&from=research.latest-advancements-list) 这有助于训练稳定性：

> [...] attention 输出门控机制有助于消除 Attention Sink 和 Massive Activation 等问题，确保整个模型的数值稳定性。


&nbsp;
## Gated DeltaNet

那么 Gated DeltaNet 到底是什么？Gated DeltaNet（Gated Delta Network 的简称）是 Qwen3-Next 的线性注意力层，目的是作为标准 softmax attention 的替代方案。正如前文提到的，它来自 [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464) 论文。

Gated DeltaNet 最初是作为 Mamba2 的改进版提出的——它把 Mamba2 的门控衰减机制和 delta rule 结合了起来。

Mamba 是一个状态空间模型（transformer 的替代方案），这个大话题值得以后单独写一篇文章来介绍。

delta rule 部分是指：计算新值与预测值之间的差（delta, Δ），用来更新一个作为记忆状态使用的隐藏状态（后面会详述）。

（题外话：熟悉经典机器学习文献的读者可以把它类比为 Hebbian learning——"一起激活的神经元会连接在一起"。它本质上可以看作感知机更新规则和基于梯度下降的学习规则的前身，不过没有监督信号。）

Gated DeltaNet 中的门控与前面讨论的 gated attention 中的门控类似，只是它用 SiLU 激活而不是 logistic sigmoid，如下图所示。（选 SiLU 大概是为了在标准 sigmoid 之上提供更好的梯度流和稳定性。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/02.webp" alt="Gated DeltaNet" width=500px>

不过，如上图所示，"gated" 中的 Gated DeltaNet 还指下面这些附加门控：

- `α`（decay gate）控制记忆随时间衰减或重置的速度；
- `β`（update gate）控制新输入更新状态的强度。

代码方面，上面 Gated DeltaNet 图示的一个简化版（不含卷积混合）大致实现如下（代码参考了 [Qwen 团队官方实现](https://github.com/huggingface/transformers/blob/0ed6d51ae8ed3f4fafca67a983b8d75bc76cd51b/src/transformers/models/qwen3_next/modular_qwen3_next.py#L835)）。

（注意：有些实现里把 decay gate 称为 `gk`（第 k 步的门控），其中 `exp(gk)` 对应论文里的 $\alpha_t$。为了把这个关系说清楚，下面的代码片段把对数空间的 gate `alpha_log` 与指数化后的 decay `alpha` 分开。）


```python
import torch
from torch import nn
import torch.nn.functional as F

def l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

class GatedDeltaNet(nn.Module):
    def __init__(
        self, d_in, d_out, dropout, num_heads, qkv_bias=False
    ):
        super().__init__()
        assert d_out % num_heads == 0

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        ####################################################
        ### NEW: Gates for delta rule and output gating
        self.W_gate = nn.Linear(d_in, d_out, bias=False)
        self.W_beta = nn.Linear(d_in, d_out, bias=False)

        # Note: The decay gate alpha corresponds to
        # A_log + W_alpha(x) + dt_bias
        self.W_alpha = nn.Linear(d_in, num_heads, bias=False)
        self.dt_bias = nn.Parameter(torch.ones(num_heads))
        A_init = torch.empty(num_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        # We could implement this as
        # W_alpha = nn.Linear(d_in, num_heads, bias=True)
        # but the bias is separate for interpretability and
        # to mimic the official implementation

        self.norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        ####################################################

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        ####################################################
        ### NEW: Compute delta rule gates
        beta = torch.sigmoid(self.W_beta(x))
        alpha_log = -self.A_log.exp().view(1, 1, -1) * F.softplus(
            self.W_alpha(x) + self.dt_bias
        )
        alpha = alpha_log.exp()
        gate = self.W_gate(x)
        ####################################################

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        beta = beta.view(b, num_tokens, self.num_heads, self.head_dim)
        gate = gate.view(b, num_tokens, self.num_heads, self.head_dim)  # NEW

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        beta = beta.transpose(1, 2)

        ####################################################
        ### NEW: QKNorm-like normalization for delta rule
        queries = l2norm(queries, dim=-1) / (self.head_dim ** 0.5)
        keys = l2norm(keys, dim=-1)
        ####################################################

        S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)

        outs = []
        ####################################################
        ### NEW: Gated delta rule update
        for t in range(num_tokens):
            k_t = keys[:, :, t]
            q_t = queries[:, :, t]
            v_t = values[:, :, t]
            b_t = beta[:, :, t]
            a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

            S = S * a_t
            kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * b_t
            S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
            ####################################################
            outs.append(y_t)

        context = torch.stack(outs, dim=2).transpose(1, 2).contiguous()
        context = context.view(b, num_tokens, self.num_heads, self.head_dim)

        ####################################################
        ### NEW: Apply RMSNorm and SiLU gate
        context = self.norm(context)
        context = context * F.silu(gate)
        ####################################################

        context = context.view(b, num_tokens, self.d_out)
        context = self.dropout(context)
        out = self.out_proj(context)
        return out
```

（注意：为了简化，我省略了 Qwen3-Next 和 Kimi Linear 用到的卷积混合，让代码更易读、聚焦在循环部分。）

所以如上所示，它和标准的（或带门控的）attention 差异很大。

在 gated attention 中，模型在所有 token 之间计算普通 attention（每个 token 都 attend 或关注其它每个 token）。然后，在得到 attention 输出之后，一个门控（sigmoid）决定保留多少输出。重点是：它仍然是常规的 scaled-dot product attention，相对上下文长度按二次复杂度扩展。

复习一下，scaled-dot product attention 算的是 softmax(QKᵀ)V，其中 Q 和 K 是 *n*×*d* 矩阵，*n* 是输入 token 数，*d* 是 embedding 维度。所以 QKᵀ 得到一个 *n*×*n* 的 attention 矩阵，再乘以一个 *n*×*d* 的 value 矩阵 V：

```
attn_scores = queries @ keys.transpose(2, 3)

mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
attn_scores.masked_fill_(
    mask_bool, torch.finfo(attn_scores.dtype).min
)

attn_weights = torch.softmax(
    attn_scores / (self.head_dim ** 0.5), dim=-1
)

context = (attn_weights @ values).transpose(1, 2)
context = context.reshape(b, num_tokens, self.d_out)
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/03.webp" alt="Quadratic attention" width=500px />

在 Gated DeltaNet 中，没有那个 *n*×*n* 的 attention 矩阵。模型逐 token 处理，保留一份运行记忆（state），每读入一个新 token 就更新一次。这在代码里就是上面那样——`S` 就是每一步 *t* 循环更新的 state。

```python
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
outs = []

for t in range(num_tokens):
    k_t = keys[:, :, t]
    q_t = queries[:, :, t]
    v_t = values[:, :, t]
    b_t = beta[:, :, t]
    a_t = alpha[:, t].unsqueeze(-1).unsqueeze(-1)

    S = S * a_t
    kv_mem = (S * k_t.unsqueeze(-1)).sum(dim=-2)
    delta = (v_t - kv_mem) * b_t
    S = S + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
    y_t = (S * q_t.unsqueeze(-1)).sum(dim=-2)
```

门控负责控制这段记忆如何变化：

- α（`alpha`）控制遗忘多少旧记忆（decay）。

- β（`beta`）控制当前时间步 *t* 的 token 更新记忆的强度。

（最后的输出 gate 在上面的代码片段里没有出现，它和 gated attention 里的类似，控制保留多少输出。）

所以从某种意义上说，Gated DeltaNet 中的 state 更新和 RNN 的工作方式类似。优点是它能随上下文长度线性扩展（通过 for 循环），而不是二次。

这种循环 state 更新的缺点是：相比普通（或带门控的）attention，它牺牲了由全配对 attention 带来的全局上下文建模能力。

Gated DeltaNet 在一定程度上仍然能捕获上下文，但必须通过记忆（*S*）这个瓶颈。记忆大小是固定的，因此效率也更高，但和 RNN 类似，它把过去上下文压缩到了一个固定大小的隐藏状态里。

这就是为什么 Qwen3-Next 和 Kimi Linear 架构不是把所有 attention 层都换成 DeltaNet 层，而是采用前面提到的 3:1 比例。

&nbsp;
## DeltaNet 节省的内存（DeltaNet Memory Savings）

前面我们讲了 DeltaNet 相比全 attention 的优势：相对上下文长度是线性而不是二次计算复杂度。

除了线性计算复杂度，DeltaNet 还有一个很大的优势——内存上的节省：DeltaNet 模块不会让 KV cache 增长。（关于 KV cache 的更多信息，参见 [../03_kv-cache](../03_kv-cache)）。如前所述，它们维护的是一个固定大小的循环 state，因此内存随上下文长度保持常量。

对于普通多头注意力（MHA）层，我们可以这样算 KV cache 大小：

```
KV_cache_MHA ≈ batch_size × n_tokens × n_heads × d_head × 2 × bytes
```

（其中的 2 倍是因为我们要存 cache 中的 key 和 value。）

对于上面这个简化版 DeltaNet，我们有：

```
KV_cache_DeltaNet = batch_size × n_heads × d_head × d_head × bytes
```

注意 `KV_cache_DeltaNet` 的大小没有上下文长度（`n_tokens`）这一项。并且我们这里只存状态 S，而不是分别保存 key 和 value，所以 `2 × bytes` 变成了 `bytes`。不过现在出现了 `d_head × d_head` 的二次项，它来自状态：

```
S = x.new_zeros(b, self.num_heads, self.head_dim, self.head_dim)
```

但这一般不用担心，因为 head 维度通常相对较小。比如 Qwen3-Next 里是 128。

加上卷积混合的完整版本会更复杂一些，还要考虑 kernel size 等因素，但上面的公式已经能说明 Gated DeltaNet 的主要趋势和动机。

可以用下面这个辅助脚本可视化不同上下文长度下的内存估算与节省：

```bash
uv run plot_memory_estimates_gated_deltanet.py \
  --emb_dim 2048 \
  --n_heads 16 \
  --n_layers 48 \
  --dtype "bf16"
```

注意上面的脚本把 `head_dim` 算成 `emb_dim / n_heads`。也就是 2048 / 16 = 128。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/gated_deltanet/plot.webp" alt="Gated DeltaNet scaling" width=500px>