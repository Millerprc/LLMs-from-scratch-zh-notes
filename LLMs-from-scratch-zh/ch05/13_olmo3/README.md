# 从零实现 Olmo 3 7B 和 32B

本目录中的 [standalone-olmo3.ipynb](standalone-olmo3.ipynb) Jupyter notebook 包含 Olmo 3 7B 和 32B 的从零实现，运行大约需要 13 GB 内存。

另一种方案 [standalone-olmo3-plus-kvcache.ipynb](standalone-olmo3-plus-kv-cache.ipynb) notebook 添加了 KV cache 以获得更好的运行时性能（但会增加代码复杂度）。了解更多关于 KV caching 的内容，请参考 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms) 这篇文章。

下面是与之并排对比的 Qwen3 作为参考模型；如果你对 Qwen3 0.6B 的 standalone notebook 感兴趣，可以从[这里](../11_qwen3)找到。

<br>

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-7B.webp?1">

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-32B.webp?1">

Olmo 3 也有不同的版本，如下图所示（架构相同，只有训练流程不同）：

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/olmo3/olmo3-pipeline.webp?1">


&nbsp;
## Olmo 3 与 Qwen3 的对比

聚焦于架构而非训练细节，本节简要对比 Olmo 3 与 Qwen3。

7B 模型：

1. 如上面的图所示，Olmo 3 架构与 Qwen3 相对相似。但需要注意的是，这种相似很可能是受 Olmo 2 前作启发，而非 Qwen3。

2) 与 Olmo 2 类似，Olmo 3 仍然使用 post-norm（而非 pre-norm），因为他们在 Olmo 2 论文中发现这能让训练更稳定。

3) 有意思的是，7B 模型仍然使用与 Olmo 2 类似的多头注意力。
   不过，为了提高效率并减小 KV cache 体积，他们现在使用滑动窗口注意力（例如，类似于 Gemma 3）。

接下来是 32B 模型：

4) 总体上是同样的架构，只是规模更大。同时，各部分比例（例如从输入到中间维度的扩展比例等）大致与 Qwen3 相符。

5) 我的猜测是，该架构最初由于词表较小而略小于 Qwen3，随后他们把中间维度的扩展从 Qwen3 的 5x 提升到了 Olmo 3 的 5.4x，以得到一个 32B 模型用于直接对比。

6) 同样值得注意的是，32B 模型（终于！）使用了 grouped query attention。

<br>

了解更多架构差异和与其他架构的对比，请参考 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) 这篇文章。