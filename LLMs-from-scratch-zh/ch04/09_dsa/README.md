# DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）

这份 bonus 资料实现了 [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2) 中提出的 DeepSeek 稀疏注意力（DeepSeek Sparse Attention, DSA）机制，首次发布于实验性的 [DeepSeek-V3.2-Exp](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp) 模型。

下面的概览依照 [From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates](https://magazine.sebastianraschka.com/p/technical-deepseek) 中关于 DSA 的讨论。

&nbsp;
## 引言（Introduction）

标准的因果 self-attention 让每个 query 关注所有前文 token，因此对长度为 L 的序列，计算的复杂度是 O(L²)，KV cache 也按 O(L) 增长。

[滑动窗口注意力（SWA）](../06_swa) 已经展示过，把 attention 限制在一个固定的局部窗口内能显著降低这个开销。在 SWA 中，每个 query token 只关注附近一段局部的前文 token。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/09.png" alt="Sliding window attention" width="800px" />

*图 1. 滑动窗口注意力把每个 query token 约束在固定的局部上下文窗口内。*

&nbsp;

DSA 沿用了"只关注前文 token 的一个子集"这一思路。但它用一个学习到的选择机制替代了固定窗口。对每个 query token，模型对候选的前文 token 打分，只保留最相关的那些。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/10.png" alt="DeepSeek Sparse Attention selected-token pattern" width="800px" />

*图 2. DeepSeek 稀疏注意力为每个 query token 学到一个前文 token 的子集。*

&nbsp;

### 架构概览（Architecture overview）

DSA 在标准 attention 的基础上加了两个组件。

**1. Lightning Indexer**

对每个 query token $t$ 和每个候选前文 token $s$，indexer 计算一个标量的相关性分数。这个实现把参考代码里的缩放因子显式写了出来：

$$I_{t,s} = \sum_{j=1}^{H_I} \frac{w_{t,j}}{\sqrt{H_I}} \cdot \text{ReLU}\left(\frac{q_{t,j} \cdot k_s}{\sqrt{d_I}}\right)$$

其中：
- $H_I$ 是轻量 indexer head 的数量，
- $q_{t,j}$ 是 token $t$ 在 head $j$ 上的 indexer query 向量，
- $k_s$ 是前文 token $s$ 的一个共享 indexer key 向量，
- $w_{t,j}$ 是一个学习到的逐头门控，缩放因子为 $1 / \sqrt{H_I}$。

ReLU 把负的点积贡献置零，逐头门控加权和则把所有 indexer head 聚合成每个前文 token 的单一相关性分数。

在完整的 DeepSeek 模型里，indexer 使用的是来自多头潜在注意力（MLA）的压缩 token 表示。本目录为了简化 GPT 实现，直接从普通隐藏状态计算 indexer 的 query 和 key。

**2. Token Selector**

在算完所有 indexer 分数之后，只保留分数最高的 top-K 个位置。所有其它位置在标准 softmax *之前* 被 mask 成 −∞，这样模型实际上只关注 $k \ll L$ 个 token。

最终稀疏性并非来自 indexer 里的 ReLU。因为分数是跨多个 indexer head 加和的，最终的分数大多仍然非零。Token selector 通过只保留 top-K 位置来形成最终的稀疏模式。

在生产级融合实现中，这能把 attention 的计算复杂度从 O(L²) 降到 O(L·k)。这里的实现保留了标准的稠密 attention 分数矩阵，在 softmax 之前应用 DSA 选出的 top-K mask。这让选择逻辑一目了然，但并没有融合 kernel 那种计算节省。

下面这张图总结了流程：lightning indexer 对候选 token 打分，selector 保留 top-K 位置，得到的 mask 限制住后续的 attention softmax。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/11.png" alt="DeepSeek Sparse Attention flowchart" width="700px" />

*图 3. DSA 先给候选 token 打分，然后保留 top-K 用于最终的 attention mask。*

&nbsp;
## 实现（Implementation）

`gpt_with_kv_dsa.py` 提供了：

| Class | Description |
|---|---|
| `LightningIndexer` | 轻量的多头打分器，给前文 token 打相关性分。 |
| `MultiHeadAttentionWithDSA` | 在标准 MHA 上加 DSA 稀疏 mask + 可选 KV cache。 |
| `GPTModel` | 兼容 GPT 风格的模型，将 MHA 替换为 `MultiHeadAttentionWithDSA`。 |

实现沿用了本仓库其它 bonus 资料的风格，可以直接当作脚本运行。它的目的是在一个小型的 GPT 风格模型里，把 DSA 机制展示得清楚。它没有实现 DeepSeek 完整的 MLA 栈、融合稀疏 kernel 或部署相关的优化。

&nbsp;
## 用法（Usage）

```bash
uv run gpt_with_kv_dsa.py \
  --emb_dim 768 \
  --n_heads 12 \
  --n_layers 12 \
  --max_new_tokens 200 \
  --index_n_heads 4 \
  --index_head_dim 64 \
  --topk 64
```

关键参数：

| Argument | Default | Description |
|---|---|---|
| `--index_n_heads` | 4 | 轻量 indexer head 数量（H_I）。 |
| `--index_head_dim` | 64 | 每个 indexer head 的维度。 |
| `--topk` | 64 | 每个 query 关注的 token 数（k）。当序列较短时上限为序列长度。 |

&nbsp;
## 与 DeepSeek V3.2 的关系（Relation to DeepSeek V3.2）

完整规模的 DeepSeek-V3.2 模型在 DSA 之外还使用了多头潜在注意力（MLA，参见 [../05_mla](../05_mla)），indexer 的 query 来自共享的压缩 latent 表示，而不是原始输入。DeepSeek-V3.2 和首次引入并试验 DSA 的 DeepSeek-V3.2-Exp 使用的是同一套架构。

关键的"挑选"思路在这里得到了复现：一个低成本的、可学习的点积打分器在 attention softmax 之前把每个 query 限制在最相关的 token 上。

图中所示的推理成本对比对于理解 DSA 在长上下文部署中为何重要是有用的背景信息。节省幅度依赖于生产级 kernel 和 serving 基础设施，所以这张图不应该被当作本目录里教学实现的 benchmark。

&nbsp;

<img src="https://sebastianraschka.com/images/blog/2025/technical-deepseek/19.png" alt="Inference cost comparison for DeepSeek Sparse Attention" width="800px" />

*图 4. DeepSeek 在长上下文 serving 中用 DSA 节省的推理成本，数据来自 [DeepSeek V3.2 技术报告](https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf)。*

&nbsp;
## 参考文献（References）

- DeepSeek V3.2 技术报告：https://huggingface.co/deepseek-ai/DeepSeek-V3.2/resolve/main/assets/paper.pdf
- DeepSeek V3.2-Exp model card 与参考代码：https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp
- Sebastian Raschka 的 "From DeepSeek V3 to V3.2: Architecture, Sparse Attention, and RL Updates"：https://magazine.sebastianraschka.com/p/technical-deepseek