# Tiny Aya 3.35B 从零实现

Tiny Aya 是 Cohere 发布的一款"小型"LLM，据称是 3B 参数规模中"最具能力的多语言开源权重模型"。（根据其[公告文章](https://cohere.com/blog/cohere-labs-tiny-aya)，Tiny Aya 在性能上优于 Qwen3-4B、Gemma 3 4B 和 Ministral 3 3B。）

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/tiny-aya/01.webp">



这是一个非常适合在本地运行和试验的模型。唯一的限制是：尽管模型开源，但其许可条款相对严格，仅允许非商业用途。

除此以外，Tiny Aya 是一款 3.35B 参数的模型，提供多个变体，适用于个人和（非商业）研究使用：

  - [tiny-aya-base](https://huggingface.co/CohereLabs/tiny-aya-base)（基础模型）
  - [tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)（跨语言和跨区域的最佳平衡，notebook 默认）
  - [tiny-aya-fire](https://huggingface.co/CohereLabs/tiny-aya-fire)（针对南亚语言优化）
  - [tiny-aya-water](https://huggingface.co/CohereLabs/tiny-aya-water)（针对欧洲和亚太地区语言优化）
  - [tiny-aya-earth](https://huggingface.co/CohereLabs/tiny-aya-earth)（针对西亚和非洲语言优化）



更具体地说，以下是各模型所优化的语言列表：

| 区域           | 语言                                                                            | 优化模型         |
| ---------------- | ------------------------------------------------------------------------------- | ----------------- |
| **亚太地区** | 繁体中文、粤语、越南语、他加禄语、爪哇语、高棉语、泰语、缅甸语、马来语、韩语、老挝语、印尼语、简体中文、日语 | tiny-aya-water  |
| **非洲**       | 祖鲁语、阿姆哈拉语、豪萨语、伊博语、斯瓦希里语、科萨语、沃洛夫语、绍纳语、约鲁巴语、尼日利亚皮钦语、马达加斯加语 |             tiny-aya-earth  |
| **南亚**   | 泰卢固语、马拉地语、孟加拉语、泰米尔语、印地语、旁遮普语、古吉拉特语、乌尔都语、尼泊尔语 | tiny-aya-fire   |
| **欧洲**       | 加泰罗尼亚语、加利西亚语、荷兰语、丹麦语、芬兰语、捷克语、葡萄牙语、法语、立陶宛语、斯洛伐克语、巴斯克语、英语、瑞典语、波兰语、西班牙语、斯洛文尼亚语、乌克兰语、希腊语、书面挪威语、罗马尼亚语、塞尔维亚语、德语、意大利语、俄语、爱尔兰语、匈牙利语、保加利亚语、克罗地亚语、爱沙尼亚语、拉脱维亚语、威尔士语 | tiny-aya-water  |
| **西亚**    | 阿拉伯语、马耳他语、土耳其语、希伯来语、波斯语                          | tiny-aya-earth  |


在架构方面，Tiny Aya 是一个经典的 decoder-style（仅解码器）Transformer，并进行了一些值得注意的改动（除了 SwiGLU 和 GQA 分组查询注意力（Grouped Query Attention）等显而易见的改动之外）：

1. **并行 Transformer 块（Parallel transformer blocks）。** 并行 Transformer 块从同一份归一化输入中并行计算 attention 和 MLP，然后将两者一次性加到残差上。我推测这样做是为了减少层内串行依赖，从而提升计算吞吐。

2. **滑动窗口注意力（Sliding window attention）。** 具体来说，它使用类似 Arcee Trinity 和 Olmo 3 的 3:1 local:global 比例。窗口大小也是 4096。此外，与 Arcee 类似，滑动窗口层使用 RoPE 旋转位置编码（Rotary Position Embedding），而全注意力层使用 NoPE（无位置编码）。

3. **LayerNorm。** 大多数架构已转向 RMSNorm，因为它计算开销更小且性能良好。Tiny Aya 仍采用更经典的方案，使用 LayerNorm 的修改版本（其实现与标准 LayerNorm 类似，但去除了 shift，即 bias 参数）。


&nbsp;
## 文件

[standalone-tiny-aya.ipynb](standalone-tiny-aya.ipynb) 是一个独立的 Jupyter notebook，实现 Tiny Aya 架构并加载预训练权重。


另一个 [standalone-tiny-aya-plus-kvcache.ipynb](standalone-tiny-aya-plus-kvcache.ipynb) notebook 在此基础上添加了 KV cache（键值缓存）以获得更好的运行时性能（但增加了更多代码复杂度）。若想进一步了解 KV 缓存，可参阅文章 [Understanding and Coding the KV Cache in LLMs from Scratch](https://magazine.sebastianraschka.com/p/coding-the-kv-cache-in-llms)。


<br>

若想深入了解架构差异，并阅读与其他架构的对比，请参阅文章 [The Big LLM Architecture Comparison: From DeepSeek-V3 to Kimi K2: A Look At Modern LLM Architecture Design](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)。