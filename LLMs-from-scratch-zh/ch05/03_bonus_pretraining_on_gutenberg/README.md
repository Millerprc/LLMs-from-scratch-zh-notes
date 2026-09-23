# 在 Project Gutenberg 数据集上预训练 GPT

本目录中的代码用于在 Project Gutenberg 提供的免费书籍上训练一个小型 GPT 模型。

正如 Project Gutenberg 网站所述，"Project Gutenberg 的电子书在美国绝大多数都属于公共领域。"

更多信息请阅读 [Project Gutenberg Permissions, Licensing and other Common Requests](https://www.gutenberg.org/policy/permission.html) 页面，了解如何使用 Project Gutenberg 提供的资源。

&nbsp;
## 如何使用这些代码

&nbsp;

### 1) 下载数据集

在本节中，我们使用 [`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) GitHub 仓库中的代码从 Project Gutenberg 下载书籍。

截至本文撰写时，这需要大约 50 GB 的磁盘空间，并花费 10-15 小时，但实际取决于自那以后 Project Gutenberg 的增长量，可能需要更长时间。

&nbsp;
#### Linux 和 macOS 用户的下载说明

Linux 和 macOS 用户可以按以下步骤下载数据集（如果你是 Windows 用户，请参阅下面的说明）：

1. 将 `03_bonus_pretraining_on_gutenberg` 文件夹设为工作目录，以便在该文件夹中本地克隆 `gutenberg` 仓库（这是运行提供的脚本 `prepare_dataset.py` 和 `pretraining_simple.py` 所必需的）。例如，当你在 `LLMs-from-scratch` 仓库的文件夹中时，通过以下命令进入 *03_bonus_pretraining_on_gutenberg* 文件夹：
```bash
cd ch05/03_bonus_pretraining_on_gutenberg
```

2. 在其中克隆 `gutenberg` 仓库：
```bash
git clone https://github.com/pgcorpus/gutenberg.git
```

3. 进入本地克隆的 `gutenberg` 仓库的文件夹：
```bash
cd gutenberg
```

4. 从 `gutenberg` 仓库文件夹中，安装 *requirements.txt* 中定义的必需包：
```bash
pip install -r requirements.txt
```

5. 下载数据：
```bash
python get_data.py
```

6. 返回到 `03_bonus_pretraining_on_gutenberg` 文件夹
```bash
cd ..
```

&nbsp;
#### Windows 用户的特殊说明

[`pgcorpus/gutenberg`](https://github.com/pgcorpus/gutenberg) 代码同时兼容 Linux 和 macOS。但是，Windows 用户需要进行一些小的调整，例如向 `subprocess` 调用添加 `shell=True` 以及替换 `rsync`。

或者，在 Windows 上运行此代码的一种更简单的方法是使用"适用于 Linux 的 Windows 子系统"（WSL）功能，该功能允许用户在 Windows 中使用 Ubuntu 运行 Linux 环境。有关更多信息，请阅读 [Microsoft 的官方安装说明](https://learn.microsoft.com/en-us/windows/wsl/install) 和 [教程](https://learn.microsoft.com/en-us/training/modules/wsl-introduction/)。

使用 WSL 时，请确保已安装 Python 3（通过 `python3 --version` 检查，或例如使用 `sudo apt-get install -y python3.10` 安装 Python 3.10），并在其中安装以下包：

```bash
sudo apt-get update && \
sudo apt-get upgrade -y && \
sudo apt-get install -y python3-pip && \
sudo apt-get install -y python-is-python3 && \
sudo apt-get install -y rsync
```

> **Note:**
> 有关如何设置 Python 和安装包的说明可在 [Optional Python Setup Preferences](../../setup/01_optional-python-setup-preferences/README.md) 和 [Installing Python Libraries](../../setup/02_installing-python-libraries/README.md) 中找到。
>
> 可选地，本仓库提供了一个运行 Ubuntu 的 Docker 镜像。有关如何使用提供的 Docker 镜像运行容器的说明可在 [Optional Docker Environment](../../setup/03_optional-docker-environment/README.md) 中找到。

&nbsp;
### 2) 准备数据集

接下来，运行 `prepare_dataset.py` 脚本，该脚本将（截至本文撰写时的 60,173 个）文本文件合并为少量较大的文件，以便更高效地传输和访问：

```bash
python prepare_dataset.py \
  --data_dir gutenberg/data/raw \
  --max_size_mb 500 \
  --output_dir gutenberg_preprocessed
```

```
...
Skipping gutenberg/data/raw/PG29836_raw.txt as it does not contain primarily English text.                                     Skipping gutenberg/data/raw/PG16527_raw.txt as it does not contain primarily English text.                                     100%|██████████████████████████████████████████████████████████| 57250/57250 [25:04<00:00, 38.05it/s]
42 file(s) saved in /Users/sebastian/Developer/LLMs-from-scratch/ch05/03_bonus_pretraining_on_gutenberg/gutenberg_preprocessed
```


> **Tip:**
> 请注意，生成的文件以明文格式存储，为了简单起见，未进行预分词。但是，如果你计划更频繁地使用该数据集或进行多轮训练，你可能希望更新代码以预分词形式存储数据集，以节省计算时间。有关更多信息，请参阅本页底部的 *Design Decisions and Improvements*。

> **Tip:**
> 你可以选择较小的文件大小，例如 50 MB。这将产生更多文件，但可能有助于在少量文件上进行更快速的预训练运行以用于测试目的。


&nbsp;
### 3) 运行预训练脚本

你可以按如下方式运行预训练脚本。请注意，附加的命令行参数以默认值显示以供说明：

```bash
python pretraining_simple.py \
  --data_dir "gutenberg_preprocessed" \
  --n_epochs 1 \
  --batch_size 4 \
  --output_dir model_checkpoints
```

输出将按以下方式格式化：

> Total files: 3
> Tokenizing file 1 of 3: data_small/combined_1.txt
> Training ...
> Ep 1 (Step 0): Train loss 9.694, Val loss 9.724
> Ep 1 (Step 100): Train loss 6.672, Val loss 6.683
> Ep 1 (Step 200): Train loss 6.543, Val loss 6.434
> Ep 1 (Step 300): Train loss 5.772, Val loss 6.313
> Ep 1 (Step 400): Train loss 5.547, Val loss 6.249
> Ep 1 (Step 500): Train loss 6.182, Val loss 6.155
> Ep 1 (Step 600): Train loss 5.742, Val loss 6.122
> Ep 1 (Step 700): Train loss 6.309, Val loss 5.984
> Ep 1 (Step 800): Train loss 5.435, Val loss 5.975
> Ep 1 (Step 900): Train loss 5.582, Val loss 5.935
> ...
> Ep 1 (Step 31900): Train loss 3.664, Val loss 3.946
> Ep 1 (Step 32000): Train loss 3.493, Val loss 3.939
> Ep 1 (Step 32100): Train loss 3.940, Val loss 3.961
> Saved model_checkpoints/model_pg_32188.pth
> Book processed 3h 46m 55s
> Total time elapsed 3h 46m 55s
> ETA for remaining books: 7h 33m 50s
> Tokenizing file 2 of 3: data_small/combined_2.txt
> Training ...
> Ep 1 (Step 32200): Train loss 2.982, Val loss 4.094
> Ep 1 (Step 32300): Train loss 3.920, Val loss 4.097
> ...


&nbsp;
> **Tip:**
> 实际上，如果你使用的是 macOS 或 Linux，我建议使用 `tee` 命令将日志输出除了打印到终端外，还保存到 `log.txt` 文件中：

```bash
python -u pretraining_simple.py | tee log.txt
```

&nbsp;
> **Warning:**
> 请注意，在 `gutenberg_preprocessed` 文件夹中 ~500 MB 文本文件中的 1 个文件上进行训练，在 V100 GPU 上大约需要 4 小时。
> 该文件夹包含 47 个文件，完成训练将需要大约 200 小时（超过 1 周）。你可能希望在较少的文件上运行它。


&nbsp;
## 设计决策与改进

请注意，此代码专注于保持简单和最小化以用于教学目的。可以通过以下方式改进代码，以提高建模性能和训练效率：

1. 修改 `prepare_dataset.py` 脚本，从每个书籍文件中去除 Gutenberg 的样板文本。
2. 更新数据准备和加载工具，以预分词数据集并以分词形式保存，这样每次调用预训练脚本时就不必重新分词。
3. 通过添加 [Appendix D: Adding Bells and Whistles to the Training Loop](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中介绍的功能（即 cosine decay、linear warmup 和 gradient clipping）来更新 `train_model_simple` 脚本。
4. 更新预训练脚本以保存优化器状态（参见第 5 章中的 *5.4 Loading and saving weights in PyTorch*；[ch05.ipynb](../../ch05/01_main-chapter-code/ch05.ipynb)），并添加加载现有模型和优化器检查点并在训练运行中断时继续训练的选项。
5. 添加更高级的日志记录器（例如 Weights and Biases）以实时查看损失和验证曲线
6. 添加分布式数据并行（DDP）并在多个 GPU 上训练模型（参见附录 A 中的 *A.9.3 Training with multiple GPUs*；[DDP-script.py](../../appendix-A/01_main-chapter-code/DDP-script.py)）。
7. 将 `previous_chapter.py` 脚本中的 from scratch `MultiheadAttention` 类替换为 [Efficient Multi-Head Attention Implementations](../../ch03/02_bonus_efficient-multihead-attention/mha-implementations.ipynb) 补充部分中实现的高效 `MHAPyTorchScaledDotProduct` 类，该类通过 PyTorch 的 `nn.functional.scaled_dot_product_attention` 函数使用 Flash Attention。
8. 通过 [torch.compile](https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html) （`model = torch.compile`）或 [thunder](https://github.com/Lightning-AI/lightning-thunder)（`model = thunder.jit(model)`）优化模型来加速训练。
9. 实现 Gradient Low-Rank Projection (GaLore) 以进一步加速预训练过程。这可以通过将 `AdamW` 优化器替换为 [GaLore Python 库](https://github.com/jiaweizzhao/GaLore) 中提供的 `GaLoreAdamW` 来实现。
