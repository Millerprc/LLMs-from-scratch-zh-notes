# 从零实现 Qwen3 与聊天界面



本附加目录包含运行一个类似 ChatGPT 的 UI 来与预训练的 Qwen3 模型交互的代码。



![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/qwen/qwen3-chainlit.gif)



为了实现这个 UI，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖

首先，我们通过 [requirements-extra.txt](requirements-extra.txt) 安装 `chainlit` 包及其依赖：

```bash
pip install -r requirements-extra.txt
```

如果使用 `uv`：

```bash
uv pip install -r requirements-extra.txt
```

&nbsp;

## 第 2 步：运行 `app` 代码

本目录包含两个文件：

1. [`qwen3-chat-interface.py`](qwen3-chat-interface.py)：该文件在 thinking 模式下加载并使用 Qwen3 0.6B 模型。
2. [`qwen3-chat-interface-multiturn.py`](qwen3-chat-interface-multiturn.py)：与上面相同，但配置为保留消息历史。

（打开并查看这些文件以了解更多细节。）

在终端运行以下其中一条命令以启动 UI 服务：

```bash
chainlit run qwen3-chat-interface.py
```

如果使用 `uv`：

```bash
uv run chainlit run qwen3-chat-interface.py
```

执行上述命令之一后会自动打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端输出，将本地地址复制到浏览器地址栏（通常是 `http://localhost:8000`）。