# 第 5 章：在无标签数据上进行预训练

### 主章节代码

- [ch05.ipynb](ch05.ipynb) 包含本章展示的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含了前几章的 `MultiHeadAttention` 模块和 `GPTModel` 类，[ch05.ipynb](ch05.ipynb) 通过 import 它们来预训练 GPT 模型
- [gpt_download.py](gpt_download.py) 包含下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习题解答

### 可选代码

- [gpt_train.py](gpt_train.py) 是一个独立的 Python 脚本文件，对应 [ch05.ipynb](ch05.ipynb) 中训练 GPT 模型所用的代码（你可以把它视为本章的脚本化总结）
- [gpt_generate.py](gpt_generate.py) 是一个独立的 Python 脚本文件，对应 [ch05.ipynb](ch05.ipynb) 中加载并使用 OpenAI 预训练模型权重的部分