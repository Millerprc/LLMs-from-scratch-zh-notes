# 附录 A：PyTorch 入门

### 主章节代码

- [code-part1.ipynb](code-part1.ipynb) 包含章节中 A.1 至 A.8 部分的全部代码
- [code-part2.ipynb](code-part2.ipynb) 包含章节中 A.9 部分关于 GPU 的全部代码
- [DDP-script.py](DDP-script.py) 包含演示多 GPU 使用的脚本（注意 Jupyter Notebook 仅支持单 GPU，所以这里采用脚本形式而非 notebook）。可通过 `python DDP-script.py` 运行。如果你的机器有 2 张以上 GPU，可通过 `CUDA_VISIBLE_DEVICES=0,1 python DDP-script.py` 运行。
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习题解答

### 可选代码

- [DDP-script-torchrun.py](DDP-script-torchrun.py) 是 `DDP-script.py` 脚本的可选版本，通过 PyTorch 的 `torchrun` 命令运行，而非通过 `multiprocessing.spawn` 自己派生并管理多个进程。`torchrun` 命令的优势在于自动处理分布式初始化，包括多节点协调，从而略微简化了设置流程。可通过 `torchrun --nproc_per_node=2 DDP-script-torchrun.py` 运行该脚本。
