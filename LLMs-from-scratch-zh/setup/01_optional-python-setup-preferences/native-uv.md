# 使用原生 uv 进行 Python 与包管理

本教程是 [README.md](./README.md) 文档中“*选项 1：使用 uv*”的替代方案，面向偏好 `uv` 原生命令而非 `uv pip` 接口的用户。`uv pip` 已经比纯 `pip` 更快，而 `uv` 原生接口比 `uv pip` 更快，因为它的开销更小，并且不需要为 PyPy 包依赖管理提供遗留兼容性。

下表比较了不同依赖与包管理方案的速度。速度对比特指安装时的包依赖解析速度，而非安装后包的运行时性能。需要注意的是，对本项目来说包的安装是一次性的，因此完全可以根据整体便利度（而不是安装速度）来选择方案。


| 命令                  | 速度对比 |
|-----------------------|-----------------|
| `conda install <pkg>` | 最慢（基准） |
| `pip install <pkg>`   | 比上面快 2-10 倍 |
| `uv pip install <pkg>`| 比上面快 5-10 倍 |
| `uv add <pkg>`        | 比上面快 2-5 倍 |

本教程重点讲解 `uv add`。


与 [README.md](./README.md) 中“*选项 1：使用 uv*”类似，本教程会带你完成基于 `uv` 的 Python 环境配置与包安装流程。

本教程基于 macOS 系统编写，但该流程对 Linux 机器同样适用，也可能适用于其他操作系统。


&nbsp;
## 1. 安装 uv

uv 可以根据你的操作系统按以下方式安装。

<br>

**macOS 与 Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者

```bash
wget -qO- https://astral.sh/uv/install.sh | sh
```

<br>

**Windows**

```bash
powershell -c "irm https://astral.sh/uv/install.ps1 | more"
```

&nbsp;

> **Note:**

> 更多安装方式请参考官方 [uv 文档](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer)。

&nbsp;
## 2. 安装 Python 包与依赖

要从 `pyproject.toml` 文件（例如本 GitHub 仓库顶层那个）安装所有必需的包，请运行以下命令，前提是该文件与你的终端会话位于同一目录：

```bash
uv sync --dev --python 3.11
```

> **注意：**
> 如果你的系统没有可用的 Python 3.11，uv 会为你下载安装。
> 建议选择比最新发布版本至少落后 1-3 个版本的 Python，以确保 PyTorch 兼容。例如，最新版本是 Python 3.13 时，建议使用 3.10、3.11、3.12。可访问 [python.org](https://www.python.org/downloads/) 查看最新 Python 版本。

> **注意：**
> 如果因为某些依赖（例如你使用的是 Windows）导致上面的命令出现问题，可以随时回退到常规 pip：
> `uv add pip`
> `uv run python -m pip install -U -r requirements.txt`


注意，上面 `uv sync` 命令会在 `.venv` 子目录中创建一个独立的虚拟环境。（如果想从头开始删除虚拟环境，只需删除 `.venv` 文件夹即可。）

你可以使用 `uv add` 安装未在 `pyproject.toml` 中声明的新包，例如：

```bash
uv add packaging
```

也可以使用 `uv remove` 移除包，例如：

```bash
uv remove packaging
```



&nbsp;
## 3. 运行 Python 代码

<br>

至此，你的环境应该已经准备好运行本仓库中的代码。

可选地，你可以运行 `python_environment_check.py` 脚本进行一次环境检查：

```bash
uv run python setup/02_installing-python-libraries/python_environment_check.py
```



<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-run-check.png?1" width="700" height="auto" alt="Uv install">


<br>

**启动 JupyterLab**

你可以通过以下命令启动 JupyterLab：

```bash
uv run jupyter lab
```

**省略 `uv run` 前缀**

如果觉得每次都要敲 `uv run` 很繁琐，可以按照下文手动激活虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows（PowerShell）上：

```bash
.venv\Scripts\activate
```

随后你就可以直接运行脚本：

```bash
python script.py
```

并启动 JupyterLab：

```bash
jupyter lab
```

&nbsp;
> **注意：**
> 如果 `jupyter lab` 命令有问题，也可以使用完整路径启动。例如，Linux/macOS 用 `.venv/bin/jupyter lab`，Windows 用 `.venv\Scripts\jupyter-lab`。

&nbsp;


&nbsp;

## 可选：手动管理虚拟环境

你也可以直接通过 `uv pip install` 从仓库安装依赖。但请注意，它不会把依赖写入 `uv.lock` 文件，而 `uv add` 会。同时它需要手动创建并激活虚拟环境：

<br>

**1. 创建新的虚拟环境**

运行以下命令手动创建一个新的虚拟环境，虚拟环境会保存在新建的 `.venv` 子目录中：

```bash
uv venv --python=python3.10
```

<br>

**2. 激活虚拟环境**

接下来需要激活这个新创建的虚拟环境。

在 macOS/Linux 上：

```bash
source .venv/bin/activate
```

在 Windows（PowerShell）上：

```bash
.venv\Scripts\activate
```

<br>

**3. 安装依赖**

最后，可以使用 `uv pip` 接口从远程源安装依赖：

```bash
uv pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```

---

有任何问题，欢迎到[讨论区](https://github.com/rasbt/LLMs-from-scratch/discussions)留言。