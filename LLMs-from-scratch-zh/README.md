# 从零开始构建大语言模型（Build a Large Language Model From Scratch）

本仓库包含用于开发、预训练和微调类 GPT 大语言模型（LLM）的代码，是书籍 [Build a Large Language Model (From Scratch)](https://amzn.to/4fqvn0D) 的官方代码仓库。

<br>
<br>

<a href="https://amzn.to/4fqvn0D"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/cover.jpg?123" width="250px"></a>

<br>

在 [*Build a Large Language Model (From Scratch)*](http://mng.bz/orYv) 一书中，你将通过一步步从零开始构建的方式，从内部理解大语言模型（LLM）的工作原理。在这本书中，我会带你创建你自己的 LLM，并通过清晰的文字、图表和示例解释每一个阶段。

本书中描述的用于训练和开发小型但功能完整的模型（用于教学目的）的方法，与构建 ChatGPT 等大规模基础模型时使用的方法一致。此外，本书还包含用于加载更大预训练模型权重以进行微调的代码。

- 官方[源代码仓库](https://github.com/rasbt/LLMs-from-scratch)链接
- [Manning 出版社网站上的书页](http://mng.bz/orYv)链接
- [Amazon.com 上的书页](https://www.amazon.com/gp/product/1633437167)链接
- ISBN 9781633437166

<a href="http://mng.bz/orYv#reviews"><img src="https://sebastianraschka.com//images/LLMs-from-scratch-images/other/reviews.png" width="220px"></a>


<br>
<br>

要下载本仓库的副本，请点击 [Download ZIP](https://github.com/rasbt/LLMs-from-scratch/archive/refs/heads/main.zip) 按钮，或在终端中执行以下命令：

```bash
git clone --depth 1 https://github.com/rasbt/LLMs-from-scratch.git
```

<br>

（如果你从 Manning 网站下载了代码包，请访问 GitHub 上的官方代码仓库 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) 以获取最新更新。）

<br>
<br>


# 目录

请注意，本 `README.md` 是一个 Markdown（`.md`）文件。如果你从 Manning 网站下载了本代码包，并在本地计算机上查看，建议使用 Markdown 编辑器或预览器以获得最佳显示效果。如果你还没有安装 Markdown 编辑器，[Ghostwriter](https://ghostwriter.kde.org) 是一个不错的免费选择。

你也可以在 GitHub 上浏览本文件及其他文件：[https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)，浏览器会自动渲染 Markdown。

<br>
<br>


> **提示：**
> 如果你正在寻找有关安装 Python 与 Python 包、以及设置代码环境的指南，建议阅读 [setup](setup) 目录中的 [README.md](setup/README.md) 文件。

<br>
<br>

[![Code tests Linux](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-linux-uv.yml)
[![Code tests Windows](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-windows-uv-pip.yml)
[![Code tests macOS](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml/badge.svg)](https://github.com/rasbt/LLMs-from-scratch/actions/workflows/basic-tests-macos-uv.yml)

- [故障排查指南](./troubleshooting.md)


| 章节标题                                              | 主代码（快速访问）                                                                                                    | 全部代码 + 补充材料      |
|------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------|
| [环境配置建议](setup) <br/>[如何最佳地阅读本书](https://sebastianraschka.com/blog/2025/reading-books.html)                            | -                                                                                                                               | -                             |
| 第 1 章：理解大语言模型                  | 无代码                                        | -                             |
| 第 2 章：处理文本数据                               | - [ch02.ipynb](ch02/01_main-chapter-code/ch02.ipynb)<br/>- [dataloader.ipynb](ch02/01_main-chapter-code/dataloader.ipynb)（摘要）<br/>- [exercise-solutions.ipynb](ch02/01_main-chapter-code/exercise-solutions.ipynb)               | [./ch02](./ch02)            |
| 第 3 章：编写注意力机制                          | - [ch03.ipynb](ch03/01_main-chapter-code/ch03.ipynb)<br/>- [multihead-attention.ipynb](ch03/01_main-chapter-code/multihead-attention.ipynb)（摘要） <br/>- [exercise-solutions.ipynb](ch03/01_main-chapter-code/exercise-solutions.ipynb)| [./ch03](./ch03)             |
| 第 4 章：从零实现一个 GPT 模型                | - [ch04.ipynb](ch04/01_main-chapter-code/ch04.ipynb)<br/>- [gpt.py](ch04/01_main-chapter-code/gpt.py)（摘要）<br/>- [exercise-solutions.ipynb](ch04/01_main-chapter-code/exercise-solutions.ipynb) | [./ch04](./ch04)           |
| 第 5 章：在无标签数据上预训练                        | - [ch05.ipynb](ch05/01_main-chapter-code/ch05.ipynb)<br/>- [gpt_train.py](ch05/01_main-chapter-code/gpt_train.py)（摘要） <br/>- [gpt_generate.py](ch05/01_main-chapter-code/gpt_generate.py)（摘要） <br/>- [exercise-solutions.ipynb](ch05/01_main-chapter-code/exercise-solutions.ipynb) | [./ch05](./ch05)              |
| 第 6 章：文本分类微调                   | - [ch06.ipynb](ch06/01_main-chapter-code/ch06.ipynb)  <br/>- [gpt_class_finetune.py](ch06/01_main-chapter-code/gpt_class_finetune.py)  <br/>- [exercise-solutions.ipynb](ch06/01_main-chapter-code/exercise-solutions.ipynb) | [./ch06](./ch06)              |
| 第 7 章：指令微调                    | - [ch07.ipynb](ch07/01_main-chapter-code/ch07.ipynb)<br/>- [gpt_instruction_finetuning.py](ch07/01_main-chapter-code/gpt_instruction_finetuning.py)（摘要）<br/>- [ollama_evaluate.py](ch07/01_main-chapter-code/ollama_evaluate.py)（摘要）<br/>- [exercise-solutions.ipynb](ch07/01_main-chapter-code/exercise-solutions.ipynb) | [./ch07](./ch07)  |
| 附录 A：PyTorch 简介                        | - [code-part1.ipynb](appendix-A/01_main-chapter-code/code-part1.ipynb)<br/>- [code-part2.ipynb](appendix-A/01_main-chapter-code/code-part2.ipynb)<br/>- [DDP-script.py](appendix-A/01_main-chapter-code/DDP-script.py)<br/>- [exercise-solutions.ipynb](appendix-A/01_main-chapter-code/exercise-solutions.ipynb) | [./appendix-A](./appendix-A) |
| 附录 B：参考文献与延伸阅读                 | 无代码                                                                                                                         | [./appendix-B](./appendix-B) |
| 附录 C：习题解答                             | - [习题解答列表](appendix-C)                                                                 | [./appendix-C](./appendix-C) |
| 附录 D：为训练循环锦上添花 | - [appendix-D.ipynb](appendix-D/01_main-chapter-code/appendix-D.ipynb)                                                          | [./appendix-D](./appendix-D)  |
| 附录 E：使用 LoRA 进行参数高效微调       | - [appendix-E.ipynb](appendix-E/01_main-chapter-code/appendix-E.ipynb)                                                          | [./appendix-E](./appendix-E) |

<br>
&nbsp;

下图是本书所涵盖内容的思维导图（mental model）。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/mental-model.jpg" width="650px">


<br>
&nbsp;

## 先决条件

最重要的先决条件是扎实的 Python 编程基础。有了这些知识，你将能更好地探索令人着迷的 LLM 世界，理解本书中介绍的概念与代码示例。

如果你有一些深度神经网络方面的经验，可能会对某些概念更为熟悉，因为 LLM 正是建立在这些架构之上的。

本书使用 PyTorch 从零实现代码，而不使用任何外部 LLM 库。虽然精通 PyTorch 不是先决条件，但熟悉 PyTorch 基础当然会很有帮助。如果你是 PyTorch 新手，附录 A 提供了简明的 PyTorch 入门介绍。另外，你也可以参考我的另一本书 [PyTorch in One Hour: From Tensors to Training Neural Networks on Multiple GPUs](https://sebastianraschka.com/teaching/pytorch-1h/) 来了解相关要点。



<br>
&nbsp;

## 硬件要求

本书主章节中的代码被设计为可在常规笔记本电脑上、在合理的时间内运行，而不需要专门的硬件。这种方式确保了广泛的读者都能接触这些材料。此外，代码在有 GPU 可用时会自动利用 GPU。（其他设备相关建议请参见 [setup](https://github.com/rasbt/LLMs-from-scratch/blob/main/setup/README.md) 文档。）


&nbsp;
## 视频课程

[一段 17 小时 15 分钟的配套视频课程](https://www.manning.com/livevideo/master-and-build-large-language-models)，我在课程中针对书中每一章进行代码讲解。课程按章节与节进行组织，与书的结构一致，因此既可以作为本书的独立替代，也可以作为配合书本一起使用的跟练资源。

<a href="https://www.manning.com/livevideo/master-and-build-large-language-models"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/video-screenshot.webp?123" width="350px"></a>


&nbsp;


## 配套书籍 / 续作

[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 虽然是一本独立书籍，但可以视为 *Build A Large Language Model (From Scratch)* 的续作。

它从一个预训练模型开始，实现不同的推理方法，包括推理时计算扩展（inference-time scaling）、强化学习与蒸馏，以提升模型的推理能力。

与 *Build A Large Language Model (From Scratch)* 类似，[*Build A Reasoning Model (From Scratch)*](https://mng.bz/lZ5B) 也采用了从零实现这些方法的动手实践方式。

<a href="https://mng.bz/lZ5B"><img src="https://sebastianraschka.com/images/reasoning-from-scratch-images/cover.webp?123" width="120px"></a>

- [Amazon 链接](https://amzn.to/4aAKiFY)
- [Manning 链接](https://mng.bz/lZ5B)
- [GitHub 仓库](https://github.com/rasbt/reasoning-from-scratch)

<br>

&nbsp;
## 练习

每章都包含若干练习。解答汇总在附录 C 中，对应的代码 notebook 可在本仓库的主章节文件夹中找到（例如，[./ch02/01_main-chapter-code/exercise-solutions.ipynb](./ch02/01_main-chapter-code/exercise-solutions.ipynb)）。

除代码练习外，你还可以从 Manning 网站下载一份免费的 170 页 PDF：[Test Yourself On Build a Large Language Model (From Scratch)](https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch)。它包含每章约 30 道测验题与解答，帮助你检验自己的理解。

<a href="https://www.manning.com/books/test-yourself-on-build-a-large-language-model-from-scratch"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/test-yourself-cover.jpg?123" width="150px"></a>

&nbsp;
## 补充材料

下面一些文件夹包含可选的补充材料，供感兴趣的读者参考：
- **环境配置**
  - [Python 环境配置建议](setup/01_optional-python-setup-preferences)
  - [安装本书使用的 Python 包与库](setup/02_installing-python-libraries)
  - [Docker 环境配置指南](setup/03_optional-docker-environment)

- **第 2 章：处理文本数据**
  - [从零实现字节对编码（BPE）分词器](ch02/05_bpe-from-scratch/bpe-from-scratch-simple.ipynb)
  - [比较不同的字节对编码（BPE）实现](ch02/02_bonus_bytepair-encoder)
  - [理解嵌入层与线性层的区别](ch02/03_bonus_embedding-vs-matmul)
  - [用简单的数字理解 Dataloader](ch02/04_bonus_dataloader-intuition)
  - [从零实现 BPE](ch02/05_bpe-from-scratch)
  - [SimpleTokenizerV3 变体](ch02/06_bonus_simple-tokenizer-v3)

- **第 3 章：编写注意力机制**
  - [比较高效多头注意力的实现](ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb)
  - [理解 PyTorch Buffers](ch03/03_understanding-buffers/understanding-buffers.ipynb)

- **第 4 章：从零实现 GPT 模型**
  - [FLOPs 分析](ch04/02_performance-analysis/flops-analysis.ipynb)
  - [KV Cache](ch04/03_kv-cache)
  - [注意力的替代方案](ch04/#attention-alternatives)
    - [分组查询注意力（Grouped-Query Attention）](ch04/04_gqa)
    - [多头潜在注意力（Multi-Head Latent Attention）](ch04/05_mla)
    - [滑动窗口注意力（Sliding Window Attention）](ch04/06_swa)
    - [门控 DeltaNet（Gated DeltaNet）](ch04/08_deltanet)
    - [DeepSeek 稀疏注意力（DeepSeek Sparse Attention，DSA）](ch04/09_dsa)
    - [跨层 KV 共享（Cross-Layer KV Sharing）](ch04/10_kv-sharing)
  - [混合专家（Mixture-of-Experts，MoE）](ch04/07_moe)

- **第 5 章：在无标签数据上预训练**
  - [替代的权重加载方法](ch05/02_alternative_weight_loading/)
  - [在 Project Gutenberg 数据集上预训练 GPT](ch05/03_bonus_pretraining_on_gutenberg)
  - [为训练循环锦上添花](ch05/04_learning_rate_schedulers)
  - [CPU 与 MPS 设备的差异](ch05/19_cpu_mps_differences)
  - [为预训练优化超参数](ch05/05_bonus_hparam_tuning)
  - [构建用户界面以与预训练 LLM 交互](ch05/06_user_interface)
  - [将 GPT 转换为 Llama](ch05/07_gpt_to_llama)
  - [内存高效加载模型权重](ch05/08_memory_efficient_weight_loading/memory-efficient-state-dict.ipynb)
  - [使用新 token 扩展 Tiktoken BPE 分词器](ch05/09_extending-tokenizers/extend-tiktoken.ipynb)
  - [加速 LLM 训练的 PyTorch 性能技巧](ch05/10_llm-training-speed)
  - [LLM 架构](ch05/#llm-architectures-from-scratch)
    - [从零实现 Llama 3.2](ch05/07_gpt_to_llama/standalone-llama32.ipynb)
    - [从零实现 Qwen3 Dense 与 Mixture-of-Experts（MoE）](ch05/11_qwen3/)
    - [从零实现 Gemma 3](ch05/12_gemma3/)
    - [从零实现 Olmo 3](ch05/13_olmo3/)
    - [从零实现 Tiny Aya](ch05/15_tiny-aya/)
    - [从零实现 Qwen3.5](ch05/16_qwen3.5/)
    - [从零实现 Gemma 4 E2B 与 E4B](ch05/17_gemma4/)
  - [第 5 章使用其他 LLM 作为即插即用替代（例如 Llama 3、Qwen 3）](ch05/14_ch05_with_other_llms/)
- **第 6 章：分类微调**
  - [微调不同层与使用更大模型的额外实验](ch06/02_bonus_additional-experiments)
  - [在 5 万条 IMDb 影评数据集上微调不同模型](ch06/03_bonus_imdb-classification)
  - [构建用户界面以与基于 GPT 的垃圾邮件分类器交互](ch06/04_user_interface)
- **第 7 章：指令微调**
  - [用于查找近似重复与生成被动语条目的数据集工具](ch07/02_dataset-utilities)
  - [使用 OpenAI API 与 Ollama 评估指令响应](ch07/03_model-evaluation)
  - [为指令微调生成数据集](ch07/05_dataset-generation/llama3-ollama.ipynb)
  - [改进用于指令微调的数据集](ch07/05_dataset-generation/reflection-gpt4.ipynb)
  - [使用 Llama 3.1 70B 与 Ollama 生成偏好数据集](ch07/04_preference-tuning-with-dpo/create-preference-data-ollama.ipynb)
  - [用于 LLM 对齐的直接偏好优化（DPO）](ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb)
  - [构建用户界面以与指令微调的 GPT 模型交互](ch07/06_user_interface)

更多补充材料请见 [Reasoning From Scratch](https://github.com/rasbt/reasoning-from-scratch) 仓库：

- **Qwen3（从零实现）基础**
  - [Qwen3 源码走读](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb)
  - [优化版 Qwen3](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02/03_optimized-LLM)

- **评估**
  - [基于验证器的评估（MATH-500）](https://github.com/rasbt/reasoning-from-scratch/tree/main/ch03)
  - [多项选择评估（MMLU）](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/02_mmlu)
  - [LLM 排行榜评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/03_leaderboards)
  - [LLM-as-a-Judge 评估](https://github.com/rasbt/reasoning-from-scratch/blob/main/chF/04_llm-judge)
- **推理时计算扩展（Inference Scaling）**
  - [自一致性（Self-Consistency）](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch04/01_main-chapter-code/ch04_main.ipynb)
  - [自我修正（Self-Refinement）](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch05/01_main-chapter-code/ch05_main.ipynb)

- **强化学习**（RL）
  - [从零实现 GRPO 的 RLVR](https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb)


<br>
&nbsp;

## 问题、反馈与为本仓库做贡献


欢迎任何形式的反馈，最佳渠道是通过 [Manning 论坛](https://livebook.manning.com/forum?product=raschka&page=1) 或 [GitHub Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions)。同样，如果你有任何问题或只是想和其他人交流想法，也请随时在论坛上发帖。

请注意，由于本仓库包含与纸质书相对应的代码，目前无法接受会扩展主章节内容的贡献，因为那会引入与纸质书的差异。保持一致有助于为所有人提供顺畅的体验。


&nbsp;
## 引用

如果你觉得本书或代码对你的研究有帮助，请考虑引用它。

Chicago 风格引用：

> Raschka, Sebastian. *Build A Large Language Model (From Scratch)*. Manning, 2024. ISBN: 978-1633437166.

BibTeX 条目：

```
@book{build-llms-from-scratch-book,
  author       = {Sebastian Raschka},
  title        = {Build A Large Language Model (From Scratch)},
  publisher    = {Manning},
  year         = {2024},
  isbn         = {978-1633437166},
  url          = {https://www.manning.com/books/build-a-large-language-model-from-scratch},
  github       = {https://github.com/rasbt/LLMs-from-scratch}
}
```