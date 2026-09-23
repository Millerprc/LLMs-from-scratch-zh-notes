# 构建与预训练 LLM 交互的用户界面



此补充文件夹包含运行类似 ChatGPT 的用户界面以与第 5 章中预训练的 LLM 交互的代码，如下所示。



![Chainlit UI example](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-orig.webp)



为了实现这个用户界面，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖项

首先，我们通过以下方式安装 `chainlit` 包：

```bash
pip install chainlit
```

（或者，执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 第 2 步：运行 `app` 代码

此文件夹包含 2 个文件：

1. [`app_orig.py`](app_orig.py)：此文件加载并使用 OpenAI 的原始 GPT-2 权重。
2. [`app_own.py`](app_own.py)：此文件加载并使用我们在第 5 章中生成的 GPT-2 权重。这要求你先执行 [`../01_main-chapter-code/ch05.ipynb`](../01_main-chapter-code/ch05.ipynb) 文件。

（打开并检查这些文件以了解更多信息。）

从终端运行以下命令之一以启动 UI 服务器：

```bash
chainlit run app_orig.py
```

或

```bash
chainlit run app_own.py
```

运行上述命令之一应该会打开一个新的浏览器标签，你可以在其中与模型交互。如果浏览器标签没有自动打开，请检查终端命令并将本地地址复制到浏览器地址栏中（通常是 `http://localhost:8000`）。