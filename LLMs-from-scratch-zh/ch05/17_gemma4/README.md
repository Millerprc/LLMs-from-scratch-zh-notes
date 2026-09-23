# Gemma 4

本目录包含一个独立、仅文本的 Gemma 4 notebook，基于 Gemma 3 参考 notebook 构建，并适配了稠密的 `google/gemma-4-E2B` 和 `google/gemma-4-E4B` checkpoint。

- [standalone-gemma4.ipynb](./standalone-gemma4.ipynb) 在纯 PyTorch 中实现共享的 Gemma 4 稠密架构，并通过 `CHOOSE_MODEL` 在 E2B 与 E4B 参考配置之间切换。