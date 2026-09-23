# 加载预训练权重的替代方案（Alternative Approaches to Loading Pretrained Weights）

如果 OpenAI 那边下载不到权重，这个目录提供了备选的权重加载策略。

- [weight-loading-pytorch.ipynb](weight-loading-pytorch.ipynb)：**（推荐）** 包含从 PyTorch state dict 加载权重的代码，这些 state dict 是我把原始 TensorFlow 权重转换后保存下来的

- [weight-loading-hf-transformers.ipynb](weight-loading-hf-transformers.ipynb)：包含通过 `transformers` 库从 Hugging Face Model Hub 加载权重的代码

- [weight-loading-hf-safetensors.ipynb](weight-loading-hf-safetensors.ipynb)：包含直接通过 `safetensors` 库从 Hugging Face Model Hub 加载权重的代码（跳过实例化 Hugging Face transformer 模型这一步）