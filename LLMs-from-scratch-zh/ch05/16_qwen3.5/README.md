# Qwen3.5 0.8B 从零实现

本目录包含 [Qwen/Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) 的从零实现风格代码。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/03.webp">

Qwen3.5 基于 Qwen3-Next 架构，我在文章 [Beyond Standard LLMs](https://magazine.sebastianraschka.com/p/beyond-standard-llms) 的 [2. (Linear) Attention Hybrids](https://magazine.sebastianraschka.com/i/177848019/2-linear-attention-hybrids) 一节中对其进行了更详细的介绍。

<a href="https://magazine.sebastianraschka.com/p/beyond-standard-llms"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen3.5/02.webp" width="500px"></a>

请注意，Qwen3.5 交替使用 `linear_attention`（线性注意力）和 `full_attention`（全注意力）层。  
这些 notebook 在保持完整模型流程可读性的同时，复用了 [qwen3_5_transformers.py](qwen3_5_transformers.py) 中的线性注意力构建模块——该文件包含来自 Hugging Face 的、采用 Apache 2.0 开源许可的线性注意力代码。

&nbsp;
## 文件

- [qwen3.5.ipynb](qwen3.5.ipynb)：Qwen3.5 0.8B 的主要 notebook 实现。
- [qwen3.5-plus-kv-cache.ipynb](qwen3.5-plus-kv-cache.ipynb)：同一模型，加上 KV cache 解码以提升效率。
- [qwen3_5_transformers.py](qwen3_5_transformers.py)：来自 Hugging Face Transformers 的、用于 Qwen3.5 线性注意力的一些辅助组件。