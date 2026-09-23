# 第 7 章：微调以遵循指令

本目录包含可用于模型评估的工具代码。



&nbsp;
## 使用 OpenAI API 评估指令回复


- [llm-instruction-eval-openai.ipynb](llm-instruction-eval-openai.ipynb) 使用 OpenAI 的 GPT-4 来评估指令微调模型生成的回复。它能处理以下格式的 JSON 文件：

```python
{
    "instruction": "What is the atomic number of helium?",
    "input": "",
    "output": "The atomic number of helium is 2.",               # <-- 测试集中给定的目标答案
    "model 1 response": "\nThe atomic number of helium is 2.0.", # <-- 某 LLM 的回复
    "model 2 response": "\nThe atomic number of helium is 3."    # <-- 第二个 LLM 的回复
},
```

&nbsp;
## 使用 Ollama 在本地评估指令回复

- [llm-instruction-eval-ollama.ipynb](llm-instruction-eval-ollama.ipynb) 提供了上述 notebook 的替代方案，通过 Ollama 使用本地下载的 Llama 3 模型。