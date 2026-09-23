# 分类微调的额外实验

下表补充了一些实验，用以回答关于各种设计选择的额外问题。第一行采用与主章节相同的设置，作为参考。例如：

- 对比第 1 行和第 2 行回答了问题：“训练最后一个 token 与第一个 token 的性能差异是多少？”；
- 对比第 1 行和第 3 行回答了问题：“只训练最后一层与只训练最后一个 block 的性能差异是多少？”；
- 以此类推。

&nbsp;

|      | 模型                | 权重         | 可训练 token 位置   | 可训练层       | 上下文长度                                          | 训练准确率 | 验证准确率 | 测试准确率 | 训练时长 | CPU/GPU |
| ---- | ------------------- | ----------- | ------------------ | -------------- | --------------------------------------------------- | --------- | ---------- | ---------- | -------- | ------- |
| 1    | gpt2-small (124M)   | 预训练       | last               | last_block     | 训练样本中最长长度 (120)                              | 96.63%    | 99.33%     | 95.00%     | 0.28 min | A100    |
| 2    | gpt2-small (124M)   | 预训练       | first              | last_block     | 训练样本中最长长度 (120)                              | 78.46%    | 80.54%     | 75.00%     | 0.28 min | A100    |
| 3    | gpt2-small (124M)   | 预训练       | last               | last_layer     | 训练样本中最长长度 (120)                              | 78.65%    | 79.87%     | 72.00%     | 0.25 min | A100    |
| 4    | gpt2-small (124M)   | 预训练       | last               | last_two_blocks | 训练样本中最长长度 (120)                              | 98.85%    | 98.66%     | 98.33%     | 0.33 min | A100    |
| 5    | gpt2-small (124M)   | 预训练       | last               | all            | 训练样本中最长长度 (120)                              | 99.62%    | 96.64%     | 96.67%     | 0.69 min | A100    |
| 6    | gpt2-medium (355M)  | 预训练       | last               | last_block     | 训练样本中最长长度 (120)                              | 87.50%    | 91.28%     | 84.67%     | 0.75 min | A100    |
| 7    | gpt2-large (774M)   | 预训练       | last               | last_block     | 训练样本中最长长度 (120)                              | 99.52%    | 98.66%     | 96.67%     | 1.50 min | A100    |
| 8    | gpt2-xl (1558M)     | 预训练       | last               | last_block     | 训练样本中最长长度 (120)                              | 99.81%    | 99.81%     | 98.33%     | 2.83 min | A100    |
| 9    | gpt2-xl (1558M)     | 预训练       | last               | all            | 训练样本中最长长度 (120)                              | 100.00%   | 98.66%     | 98.67%     | 8.12 min | A100    |
| 10   | gpt2-small (124M)   | 随机         | last               | all            | 训练样本中最长长度 (120)                              | 100.00%   | 96.64%     | 93.67%     | 0.69 min | A100    |
| 11   | gpt2-small (124M)   | 预训练       | last               | LoRA           | 训练样本中最长长度 (120)                              | 100.00%   | 97.32%     | 96.67%     | 0.75 min | A100    |
| 12   | gpt2-xl (1558M)     | 预训练       | last               | LoRA           | 训练样本中最长长度 (120)                              | 100.00%   | 98.66%     | 98.33%     | 5.79 min | A100    |
| 13   | gpt2-small (124M)   | 预训练       | last               | last_block     | 上下文长度 (1024)                                    | 83.08%    | 87.92%     | 78.33%     | 2.46 min | A100    |
| 14   | gpt2-small (124M)   | 预训练       | last               | last_block     | 可变：无 padding (batch size 1)                      | 100.00%   | 98.66%     | 98.00%     | 1.75 min | A100    |
| 15   | gpt2-small (124M)   | 预训练       | last               | last_block     | 可变：无 padding (batch size 8)                      | 99.33%    | 98.66%     | 98.33%     | 1.70 min | A100    |
| 16   | gpt2-small (124M)   | 预训练       | last               | last_block     | 灵活（最后一个非 padding 位置）                       | 99.42%    | 98.66%     | 98.33%     | 0.30 min | A100    |
| 17   | gpt2-small (124M)   | 预训练       | last               | last_block     | 训练样本中最长长度 (120)；但不使用 causal mask         | 99.23%    | 98.66%     | 95.33%     | 0.29 min | A100    |
| 18   | gpt2-small (124M)   | 预训练       | last               | last_block     | 训练样本中最长长度 (120)，并对 padding 使用 `ignore_index` | 96.63%    | 99.33%     | 95.00%     | 0.28 min | A100    |
| 19   | gpt2-small (124M)   | 预训练       | last + pooled embeddings | last_block | 训练样本中最长长度 (120)                              | 97.79%    | 99.33%     | 96.33%     | 0.32 min | A100    |

&nbsp;

### 用法

你可以使用以下命令重现这些实验：

- 第 1 行：`python additional_experiments.py`
- 第 2 行：`python additional_experiments.py --trainable_token_pos first`
- 第 3 行：`python additional_experiments.py --trainable_layers last_layer`
- 第 4 行：`python additional_experiments.py --trainable_layers last_two_blocks`
- 第 5 行：`python additional_experiments.py --trainable_layers all`
- 第 6 行：`python additional_experiments.py --model_size "gpt2-medium (355M)"`
- 第 7 行：`python additional_experiments.py --model_size "gpt2-large (774M)"`
- 第 8 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)"`
- 第 9 行：`python additional_experiments.py --model_size "gpt2-xl (1558M)" --trainable_layers all`
- 第 10 行：`python additional_experiments.py --weights random --trainable_layers all`
- 第 11 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 16`
- 第 12 行：`python additional_experiments.py --trainable_layers lora --lora_rank 16 --lora_alpha 8 --model_size "gpt2-xl (1558M)"`
- 第 13 行：`python additional_experiments.py --context_length "model_context_length"`
- 第 14 行：`python additional_experiments.py --no_padding --batch_size 1`
- 第 15 行：`python additional_experiments.py --no_padding --batch_size 1 --accumulation_steps 8`
- 第 16 行：`python additional_experiments.py --trainable_token_pos "flexible"`
- 第 17 行：`python additional_experiments.py --disable_causal_mask`
- 第 18 行：`python additional_experiments.py --ignore_index 50256`
- 第 19 行：`python additional_experiments.py --average_embeddings`

我刻意把 LLM 和数据集都保持较小，因此即便没有 GPU，你也可以在普通笔记本（如 MacBook Air M3）上以大约 15 分钟跑完默认设置的训练。

&nbsp;

### 结果解读

1. **训练最后一个 vs. 第一个输出 token 位置（第 1 行 vs. 第 2 行）**：训练最后一个输出 token 位置的表现明显优于训练第一个。这一改进符合 causal self-attention mask 的预期。
2. **训练最后一个 Transformer block vs. 最后一层（第 1 行 vs. 第 3 行）**：训练整个最后一个 Transformer block 的结果也明显优于仅训练最后一层。
3. **训练最后一个 vs. 最后两个 Transformer block（第 1 行 vs. 第 4 行）**：训练最后两个 Transformer block 而不仅仅是最后一个，能带来明显的 3.33% 准确率提升。
4. **训练最后一个 Transformer block vs. 全部层（第 1 行 vs. 第 5 行）**：训练所有层相比仅训练最后一个 block 能带来约 2% 的小幅提升，但训练时长几乎增加到三倍。同时其效果也不如训练 12 个 Transformer block 的最后两个。
5. **使用更大的预训练模型（第 1 行 vs. 第 6 行，以及第 1 行 vs. 第 7、8 行）**：3 倍大的预训练模型反而带来更差的结果，而使用 5 倍大的模型相对初始模型提升了表现，这在意料之中。类似地，12 倍大的模型进一步提升了预测性能。（可能 medium 模型没有充分预训练，或者该微调配置对这个模型不理想。）
6. **使用随机权重模型 vs. 预训练权重（第 1、5 行 vs. 第 10 行）**：随机权重模型的结果只比预训练权重差 3% 和 1.3%。
7. **使用 LoRA（Low-Rank Adaptation）vs 训练所有层（第 11 行 vs. 第 5 行，以及第 12 行 vs. 第 9 行）**：冻结原模型并加入可训练的 LoRA 层（细节见 [Appendix E](../../appendix-E/01_main-chapter-code/appendix-E.ipynb)）是训练所有模型参数之外的一个可行替代方案，甚至能让性能提升 1 个百分点（第 11 行 vs. 第 5 行）。从训练与验证准确率之间约 1% 的更小差距可以看出，这很可能是因为过拟合减少了。此外，使用 LoRA 也更节省显存，因为需要更新的参数更少。训练更大的模型时（第 12 行 vs. 第 9 行），可以看到 LoRA 训练速度也快得多（5.79 min 对比 8.12 min）。
8. **将输入 padding 到完整上下文长度 vs. 训练样本中最长长度（第 1 行 vs. 第 13 行）**：padding 到支持的完整上下文长度会让结果显著变差。
9. **Padding vs 不 padding（第 1 行 vs. 第 14、15、16 行）**：`--no_padding` 选项会禁用数据集中的 padding，这要求使用 batch size 为 1 来处理变长输入。这样能得到更高的测试准确率，但训练更慢。在第 15 行中，我们额外启用 8 步梯度累加，使等效 batch size 与其他实验一致，这有助于减轻过拟合并略微提升测试集准确率。在第 16 行中，我们仍然 padding，但根据最后一个非 padding token 选择 token 位置。第 16 行在数学上应与使用梯度累加的第 15 行相似。然而，由于 token 数量不一致时梯度累加存在一些挑战，结果可能会有细微出入（详见[这篇](https://unsloth.ai/blog/gradient)博文）。
10. **禁用 causal attention mask（第 1 行 vs. 第 17 行）**：禁用多头注意力模块中的 causal attention mask，这意味着所有 token 都可以关注其他所有 token。模型准确率比使用 causal mask 的 GPT 模型略有提升。
11. **在 loss 和反向传播中忽略 padding 索引（第 1 行 vs. 第 18 行）**：设置 `--ignore_index 50256` 会在 PyTorch 的 `cross_entropy` loss 中排除 `` padding tokens。这里没有可见效果，因为我们替换了输出层，使得 token ID 只取 0 或 1 以用于二分类任务。但在第 7 章对模型进行指令微调时，这个设置很有用。
12. **对所有 token 的 embedding 取平均（第 1 行 vs. 第 19 行）**：设置 `--average_embeddings` 会对所有 token 的 embedding 取平均。如果不启用该选项（默认行为），则只使用 `--trainable_token_pos` 指定的 token 位置（例如最后一个 token）的输出 embedding。启用 `--average_embeddings` 后，会对所有 token 的 embedding 做 mean pooling，再写入 `--trainable_token_pos` 指定的位置（默认最后一个）。可以看到，准确率从 95.00% 提升到 96.33%，运行时间仅小幅增加（0.28 min 到 0.32 min），实际中或许值得一试。