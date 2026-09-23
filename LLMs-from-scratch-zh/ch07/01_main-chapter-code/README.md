# 第 7 章：微调以遵循指令

### 主章节代码

- [ch07.ipynb](ch07.ipynb) 包含本章展示的所有代码
- [previous_chapters.py](previous_chapters.py) 是一个 Python 模块，包含了前几章编写并训练的 GPT 模型，以及许多工具函数，本章会复用它们
- [gpt_download.py](gpt_download.py) 包含下载预训练 GPT 模型权重的工具函数
- [exercise-solutions.ipynb](exercise-solutions.ipynb) 包含本章的练习题解答

### 可选代码

- [load-finetuned-model.ipynb](load-finetuned-model.ipynb) 是一个独立的 Jupyter notebook，用于加载本章微调得到的模型
- [gpt_instruction_finetuning.py](gpt_instruction_finetuning.py) 是一个独立的 Python 脚本文件，对应 [ch07.ipynb](ch07.ipynb) 中指令微调 GPT 模型所用的代码（你可以把它视为本章的脚本化总结）

用法：

```bash
python gpt_instruction_finetuning.py
```

```
matplotlib version: 3.9.0
tiktoken version: 0.7.0
torch version: 2.3.1
tqdm version: 4.66.4
tensorflow version: 2.16.1
--------------------------------------------------
Training set length: 935
Validation set length: 55
Test set length: 110
--------------------------------------------------
Device: cpu
--------------------------------------------------
File already exists and is up-to-date: gpt2/355M/checkpoint
File already exists and is up-to-date: gpt2/355M/encoder.json
File already exists and is up-to-date: gpt2/355M/hparams.json
File already exists and is up-to-date: gpt2/355M/model.ckpt.data-00000-of-00001
File already exists and is up-to-date: gpt2/355M/model.ckpt.index
File already exists and is up-to-date: gpt2/355M/model.ckpt.meta
File already exists and is up-to-date: gpt2/355M/vocab.bpe
Loaded model: gpt2-medium (355M)
--------------------------------------------------
Initial losses
   Training loss: 3.839039182662964
   Validation loss: 3.7619192123413088
Ep 1 (Step 000000): Train loss 2.611, Val loss 2.668
Ep 1 (Step 000005): Train loss 1.161, Val loss 1.131
Ep 1 (Step 000010): Train loss 0.939, Val loss 0.973
...
Training completed in 15.66 minutes.
Plot saved as loss-plot-standalone.pdf
--------------------------------------------------
Generating responses
100%|█████████████████████████████████████████████████████████| 110/110 [06:57<00:00,  3.80s/it]
Responses saved as instruction-data-with-response-standalone.json
Model saved as gpt2-medium355M-sft-standalone.pth
```

- [ollama_evaluate.py](ollama_evaluate.py) 是一个独立的 Python 脚本，用于对微调后模型的回复进行评估，对应主章节中的评估部分（你可以把它视为本章脚本化总结中专注于评估的部分）

用法：

```bash
python ollama_evaluate.py --file_path instruction-data-with-response-standalone.json
```

```
Ollama running: True
Scoring entries: 100%|███████████████████████████████████████| 110/110 [01:08<00:00,  1.62it/s]
Number of scores: 110 of 110
Average score: 51.75
```

- [exercise_experiments.py](exercise_experiments.py) 是一个可选脚本，实现了练习题的解法；更多细节参见 [exercise-solutions.ipynb](exercise-solutions.ipynb)