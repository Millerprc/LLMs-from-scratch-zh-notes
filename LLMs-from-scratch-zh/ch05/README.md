# 第 5 章：在无标签数据上预训练

&nbsp;
## 主章节代码

- [01_main-chapter-code](01_main-chapter-code) 包含主章节代码

&nbsp;
## 补充材料

- [02_alternative_weight_loading](02_alternative_weight_loading) 包含当模型权重无法从 OpenAI 获取时，从其他位置加载 GPT 模型权重的代码
- [03_bonus_pretraining_on_gutenberg](03_bonus_pretraining_on_gutenberg) 包含在整个 Project Gutenberg 书籍语料上对 LLM 进行更长时间预训练的代码
- [04_learning_rate_schedulers](04_learning_rate_schedulers) 包含实现更复杂训练函数的代码，包括学习率调度器（learning rate schedulers）与梯度裁剪（gradient clipping）
- [05_bonus_hparam_tuning](05_bonus_hparam_tuning) 包含一个可选的超参数调优脚本
- [06_user_interface](06_user_interface) 实现一个与预训练 LLM 交互的交互式用户界面
- [08_memory_efficient_weight_loading](08_memory_efficient_weight_loading) 包含一个补充 notebook，展示如何通过 PyTorch 的 `load_state_dict` 方法更高效地加载模型权重
- [09_extending-tokenizers](09_extending-tokenizers) 包含 GPT-2 BPE 分词器的从零实现
- [10_llm-training-speed](10_llm-training-speed) 展示提升 LLM 训练速度的 PyTorch 性能技巧
- [18_muon](18_muon) 解释如何在 GPT 模型训练设置中使用 Muon 优化器
- [19_cpu_mps_differences](19_cpu_mps_differences) 汇总 CPU 与 MPS 设备表现不同的示例

&nbsp;
## 从零实现 LLM 架构

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen-overview.webp">

&nbsp;


- [07_gpt_to_llama](07_gpt_to_llama) 包含将 GPT 架构实现逐步转换为 Llama 3.2 并加载 Meta AI 预训练权重的分步指南
- [11_qwen3](11_qwen3) 从零实现 Qwen3 0.6B 与 Qwen3 30B-A3B（混合专家，Mixture-of-Experts），包括加载 base、reasoning 与 coding 模型变体预训练权重的代码
- [12_gemma3](12_gemma3) 从零实现 Gemma 3 270M 及其带 KV cache 的替代版本，包括加载预训练权重的代码
- [13_olmo3](13_olmo3) 从零实现 Olmo 3 7B 与 32B（Base、Instruct 与 Think 变体）及其带 KV cache 的替代版本，包括加载预训练权重的代码
- [17_gemma4](17_gemma4) 从零实现 Gemma 4 的 E2B 与 E4B 密集变体

&nbsp;
## 本章跟练视频

<br>
<br>

[![视频链接](https://img.youtube.com/vi/Zar2TJv-sE0/0.jpg)](https://www.youtube.com/watch?v=Zar2TJv-sE0)