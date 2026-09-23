# 第 4 章：从零实现一个 GPT 模型以生成文本

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码。

&nbsp;
## 补充材料

- [02_performance-analysis](02_performance-analysis) 包含用于分析主章节中实现的 GPT 模型性能的可选代码
- [03_kv-cache](03_kv-cache) 实现一个 KV cache 以加速推理时的文本生成
- [07_moe](07_moe) 混合专家（Mixture-of-Experts，MoE）的解释与实现
- [ch05/07_gpt_to_llama](../ch05/07_gpt_to_llama) 包含将 GPT 架构实现逐步转换为 Llama 3.2 并加载 Meta AI 预训练权重的分步指南（在完成第 4 章之后了解替代架构可能会很有意思，但你也可以留到读完第 5 章后再看）


&nbsp;
## 注意力的替代方案

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/attention-alternatives/attention-alternatives.webp">

&nbsp;

- [04_gqa](04_gqa) 介绍分组查询注意力（Grouped-Query Attention，GQA），它被大多数现代 LLM（Llama 4、gpt-oss、Qwen3、Gemma 3 等）用作常规多头注意力（MHA）的替代方案
- [05_mla](05_mla) 介绍多头潜在注意力（Multi-Head Latent Attention，MLA），它被 DeepSeek V3 用作常规多头注意力（MHA）的替代方案
- [06_swa](06_swa) 介绍滑动窗口注意力（Sliding Window Attention，SWA），它被 Gemma 3 等使用
- [08_deltanet](08_deltanet) 介绍作为流行线性注意力变体的门控 DeltaNet（Gated DeltaNet）（被 Qwen3-Next 与 Kimi Linear 使用）
- [10_kv-sharing](10_kv-sharing) 介绍跨层 KV 共享（cross-layer KV sharing），Gemma 4 E2B 与 E4B 用它来降低 KV-cache 内存占用


&nbsp;
## 更多

在下面的视频中，我提供了一次跟练 session，覆盖本章的部分内容作为补充材料。

<br>
<br>

[![视频链接](https://img.youtube.com/vi/YSAkgEarBGE/0.jpg)](https://www.youtube.com/watch?v=YSAkgEarBGE)