# 用于加速 LLM 训练的 PyTorch 性能技巧



请注意，本书是为教育目的编写的，这意味着原始代码有意保持简单。这是为了提高可读性并确保跨不同硬件（包括 CPU 和 GPU）的兼容性。但是，你可能对一些更高级的 PyTorch 和 GPU 功能感兴趣，以使 LLM 训练性能更佳。

此文件夹包含三个代码文件，演示了第 5 章中介绍的 LLM 和训练函数的性能优化：

1. [`00_orig.py`](00_orig.py)：第 5 章的原始代码，用于 CPU 和单 GPU 训练。  
   ➤ 通过以下方式运行：`python 00_orig.py`

2. [`01_opt_single_gpu.py`](01_opt_single_gpu.py)：针对单 GPU 训练的优化版本。  
   ➤ 通过以下方式运行：`python 01_opt_single_gpu.py`

3. [`02_opt_multi_gpu_ddp.py`](02_opt_multi_gpu_ddp.py)：使用分布式数据并行（DDP）的多 GPU 训练优化版本。  
   ➤ 通过以下方式运行：`torchrun --nproc_per_node=4 02_opt_multi_gpu_ddp.py`  
   (**注意：** 为了与 `01_opt_single_gpu.py` 相比保持更改最小化，此脚本仅支持通过上述 `torchrun` 进行多进程。这意味着**不**支持通过 `python 02_opt_multi_gpu_ddp.py` 进行多 GPU 训练）

**请注意，这些修改将训练速度从每秒 12,525 个 token（单 A100）提高到每秒 142,156 个 token（单 A100）以及每秒 419,259 个 token（4 个 A100）。**

我计划在未来更详细的文章中扩展这些差异。目前，查看代码改进的最简单方法是在 Visual Studio Code 中打开文件并使用"Compare Selected"功能查看差异。

![VS compare](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/llm-training-speed/vs-code-compare.png)

![PyTorch Tips](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/pytorch-tips/pytorch-tips.webp?1)


&nbsp;
## 单 GPU 速度比较

如上所述，我计划在未来更详细地阐述这些更改。目前，本节包含以 tokens/second 为单位的每项修改的简单性能概览。所有实验均在 A100 GPU 上运行。

&nbsp;
### 基线

请注意，`00_orig.py` 作为基线，除了以下内容外，没有任何重大修改，直接使用第 5 章中的代码：

- 4 倍大的上下文长度（这解释了与第 5 章相比 `00_orig.py` 相对较大的内存占用）；
- 4 倍 batch size（这是 `00_orig.py` 相对较大内存占用的另一个原因）；
- 一个更大的公共领域书籍来增加训练数据大小。

超参数并未针对最小化损失和减少过拟合进行很好的优化，LLM 在最后生成的文本可能不是很复杂；但是，这应该不重要，因为主要要点是 `tok/sec` 指标，它在此作为速度参考（越高越好）。

```bash
ubuntu@159-13-52-60:~$ python 00_orig.py
PyTorch version: 2.6.0+cu124
Using cuda
CUDA version: 12.4

Ep 1, Step 000000, Train: 9.535, Val: 9.609, Step tok/sec: 7238, Avg tok/sec: 0
Ep 1, Step 000015, Train: 6.201, Val: 6.152, Step tok/sec: 12545, Avg tok/sec: 12545
Ep 1, Step 000030, Train: 5.663, Val: 5.688, Step tok/sec: 12490, Avg tok/sec: 12517
Ep 1, Step 000045, Train: 5.316, Val: 5.362, Step tok/sec: 12541, Avg tok/sec: 12525
Every effort moves you, and's, and I am not be a

...

Ep 15, Step 000735, Train: 0.227, Val: 6.818, Step tok/sec: 11599, Avg tok/sec: 12248
Ep 15, Step 000750, Train: 0.300, Val: 6.895, Step tok/sec: 12530, Avg tok/sec: 12253
Ep 15, Step 000765, Train: 0.150, Val: 6.914, Step tok/sec: 12532, Avg tok/sec: 12259
Every effort moves you like best to think which he held in the room in him, the interest was the night, the realities of the affairs Bulstrode's duty, now!' the fact is another man, conquests

Allocated memory: 2.5069 GB
Reserved memory: 26.2617 GB
```

请注意，`01_opt_single_gpu.py` 按顺序包含下面列出的所有修改。

比较始终基于上一节第一个 epoch 之后的平均 tok/sec 和已分配内存。

&nbsp;
### 1. 动态创建 causal mask

- 不保存 causal mask，而是动态创建 causal mask 以减少内存使用（这里效果最小，但在像支持 131k 输入 token 的 Llama 3.2 这样的长上下文大小模型中可能会累加起来）

之前：
- `Avg tok/sec: 12525`
- `Reserved memory: 26.2617 GB`

之后：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 2. 使用 tensor cores

- 使用 tensor cores（仅适用于 A100 等 Ampere GPU 及更新版本）

之前：
- `Avg tok/sec: 12526`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 3. Fused AdamW 优化器

- 通过设置 `fused=True` 使用 `AdamW` 的融合内核

之前：
- `Avg tok/sec: 27648`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 4. 数据加载器中的 pinned memory

- 在数据加载器中使用 `pin_memory=True` 以预分配和重用 GPU 内存

之前：
- `Avg tok/sec: 28399`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

&nbsp;
### 5. 使用 bfloat16 精度

- 从 32 位浮点切换到 16 位 brain float (bfloat16) 精度（有关此主题的更多信息，请参阅我的 [article here](https://magazine.sebastianraschka.com/p/the-missing-bits-llama-2-weights)）

之前：
- `Avg tok/sec: 28402`
- `Reserved memory: 26.2422 GB`

之后：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

&nbsp;
### 6. 用 PyTorch 类替换 from-scratch 代码

- 用 PyTorch 的原生实现替换 LayerNorm 和 GeLU 的 from-scratch 实现

之前：
- `Avg tok/sec: 45486`
- `Reserved memory: 13.7871 GB`

之后：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

&nbsp;
### 7. 使用 FlashAttention

- 使用 PyTorch 的 self-attention 函数配合 FlashAttention，而不是我们 from-scratch 的多头注意力实现。


之前：
- `Avg tok/sec: 55256`
- `Reserved memory: 11.5645 GB`

之后：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

&nbsp;
### 8. 使用 `pytorch.compile`

- 使用 `torch.compile(model)`。请注意，最初几次迭代总是较慢，然后才会加速。由于 `Avg tok/sec` 测量仅包括平均计算中的第一行，我们现在使用 epoch 1 末尾的 `Step tok/sec`。


之前：
- `Avg tok/sec: 91901`
- `Reserved memory: 5.9004 GB`

之后：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

<br>

---

**Windows note**

- 编译在 Windows 上可能比较棘手
- `torch.compile()` 使用 Inductor，它 JIT 编译内核，需要一个可用的 C/C++ 工具链
- 对于 CUDA，Inductor 还依赖于 Triton，可通过社区包 `triton-windows` 获得
  - 如果你看到 `cl not found`，请 [安装带有"C++ workload"的 Visual Studio Build Tools](https://learn.microsoft.com/en-us/cpp/build/vscpp-step-0-installation?view=msvc-170) 并从"x64 Native Tools"提示符运行 Python
  - 如果你使用 CUDA 时看到 `triton not found`，请安装 `triton-windows`（例如，`uv pip install "triton-windows<3.4"`）。
- 对于 CPU，一位读者进一步推荐遵循 [PyTorch Inductor guide for Windows](https://docs.pytorch.org/tutorials/unstable/inductor_windows.html)
  - 这里，重要的是在安装 Visual Studio 2022 时安装英语语言包以避免 UTF-8 错误
  - 还要注意，需要通过"Visual Studio 2022 Developer Command Prompt"而不是 notebook 来运行代码
- 如果此设置比较棘手，你可以跳过编译；**编译是可选的，所有代码示例在没有编译的情况下也能正常工作**

---

&nbsp;
### 9. Vocabulary padding

- 在这里，我们将词汇表大小从 50,257 略微增加到 50,304，这是最接近的 64 的倍数。这个技巧是我的前同事 Carlos Mocholi 建议给我的，他提到这最初来自 Andrej Karpathy（可能来自 [this post](https://x.com/karpathy/status/1621578354024677377)）。Karpathy 的建议基于与 PyTorch 团队的互动，正如 [Bertrand Maher](https://www.linkedin.com/feed/update/urn:li:activity:7309569006057795584?commentUrn=urn%3Ali%3Acomment%3A%28activity%3A7309569006057795584%2C7309754284185669632%29&dashCommentUrn=urn%3Ali%3Afsd_comment%3A%287309754284185669632%2Curn%3Ali%3Aactivity%3A7309569006057795584%29) 所提到的。关于 `torch.compile` 的建议。一个很好的资源是 [NVIDIA's guidelines on tensor shapes](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html#tensor-core-shape)，其中 batch size 和线性层维度通常选择为某些值的倍数。此外，vocab-padding 技巧由 NVIDIA 的 Megatron 团队在很久以前描述过（参见 2019 年的 [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) 论文）。

之前：
- `Step tok/sec: 112046`
- `Reserved memory: 6.1875 GB`

之后：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

&nbsp;
### 10. 增加 batch size

- 最后，我们将 batch size 增加到 GPU 支持的最大 2 的幂

之前：
- `Step tok/sec: 127345`
- `Reserved memory: 5.8906 GB`

之后：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`


&nbsp;
## 多 GPU 速度比较

由于我们现在使用 4 个 GPU 而不是 1 个，这可能不是完全公平的比较，但是，使用分布式数据并行（如果训练不受有限 GPU 内存限制，则最快的多 GPU 技术）当然可以带来显著的速度提升：

之前（单 GPU）：
- `Step tok/sec: 142156`
- `Reserved memory: 22.5078 GB`

之后（4 个 GPU）：
- `Step tok/sec: 419259`
- `Reserved memory: 22.7969 GB`