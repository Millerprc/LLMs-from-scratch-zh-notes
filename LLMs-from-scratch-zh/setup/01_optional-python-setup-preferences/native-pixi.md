# 使用原生 pixi 进行 Python 与包管理

本教程是 [README.md](./README.md) 文档中“*选项 1：使用 uv*”的替代方案，面向偏好 `pixi` 原生命令而非 `conda`、`pip` 等传统环境与包管理工具的用户。

注意，pixi 底层使用的是 `uv add`，详细说明见 [`./native-uv.md`](native-uv.md)。

pixi 与 uv 都是现代化的 Python 包与环境管理工具，但 pixi 是一个多语言（polyglot）包管理器，不仅能管理 Python，也能管理其他语言（类似于 conda）；而 uv 是专为 Python 设计的工具，专注于极快的依赖解析与包安装速度。

如果你需要一款支持多语言（不仅是 Python）的多语言包管理器，或更倾向于类似 conda 的声明式环境管理方式，可能会选择 pixi 而非 uv。更多信息请访问官方 [pixi 文档](https://pixi.sh/latest/)。

本教程基于 macOS 系统编写，但该流程同样适用于 Linux，也可能适用于其他操作系统。

&nbsp;
## 1. 安装 pixi

pixi 可以根据你的操作系统按以下方式安装。

<br>

**macOS 与 Linux**

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

或者

```bash
wget -qO- https://pixi.sh/install.sh | sh
```

<br>

**Windows**

从官方[文档](https://pixi.sh/latest/installation/#__tabbed_1_2)下载安装包，或运行其中列出的 PowerShell 命令。



> **注意：**
> 更多安装方式请参考官方 [pixi 文档](https://pixi.sh/latest/)。


&nbsp;
## 1. 安装 Python

你可以使用 pixi 安装 Python：

```bash
pixi add python=3.10
```

> **注意：**
> 建议安装比当前最新版本至少落后 2 个版本的 Python，以保证兼容性。例如，最新版本是 Python 3.13 时，建议安装 3.10 或 3.11。可访问 [python.org](https://www.python.org) 查看当前最新 Python 版本。

&nbsp;
## 3. 安装 Python 包与依赖

要从 `pixi.toml` 文件（例如本 GitHub 仓库顶层那个）安装所有必需的包，请运行以下命令，前提是该文件与你的终端会话位于同一目录：

```bash
pixi install
```

> **注意：**
> 如果遇到依赖相关的问题（例如你使用的是 Windows），随时可以回退到 pip：`pixi run pip install -U -r requirements.txt`

默认情况下，`pixi install` 会为该项目创建一个独立的虚拟环境。

你可以通过 `pixi add` 安装未在 `pixi.toml` 中声明的新包，例如：

```bash
pixi add packaging
```

也可以通过 `pixi remove` 移除包，例如：

```bash
pixi remove packaging
```

&nbsp;
## 4. 运行 Python 代码

至此，你的环境应该已经准备好运行本仓库中的代码。

可选地，你可以执行本仓库中的 `python_environment_check.py` 脚本运行一次环境检查：

```bash
pixi run python setup/02_installing-python-libraries/python_environment_check.py
```

<br>

**启动 JupyterLab**

你可以通过以下命令启动 JupyterLab：

```bash
pixi run jupyter lab
```

---

有任何问题，欢迎到[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)留言。