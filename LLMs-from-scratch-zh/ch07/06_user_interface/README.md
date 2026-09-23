# 构建一个用户界面与指令微调后的 GPT 模型交互



本附加目录包含构建一个类似 ChatGPT 的 UI 与第 7 章指令微调后的 GPT 模型进行交互的代码，如下所示。



![Chainlit UI 示例](https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/chainlit/chainlit-sft.webp?2)



为了实现这个 UI，我们使用开源的 [Chainlit Python 包](https://github.com/Chainlit/chainlit)。

&nbsp;
## 第 1 步：安装依赖

首先，通过以下命令安装 `chainlit`：

```bash
pip install chainlit
```

（也可以执行 `pip install -r requirements-extra.txt`。）

&nbsp;
## 第 2 步：运行 `app` 代码

[`app.py`](app.py) 文件包含 UI 相关代码。打开并查看这些文件以了解细节。

该文件会加载并使用我们在第 7 章生成的 GPT-2 权重，因此你需要先执行 [`../01_main-chapter-code/ch07.ipynb`](../01_main-chapter-code/ch07.ipynb)。

在终端运行以下命令以启动 UI 服务：

```bash
chainlit run app.py
```

执行上述命令后会自动打开一个新的浏览器标签页，你可以在其中与模型交互。如果浏览器标签页没有自动打开，请查看终端输出，将本地地址复制到浏览器地址栏（通常是 `http://localhost:8000`）。