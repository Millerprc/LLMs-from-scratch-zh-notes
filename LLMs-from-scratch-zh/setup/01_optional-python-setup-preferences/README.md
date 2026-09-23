# Python 设置技巧



安装 Python 和配置计算环境有多种方式。在这里,我分享我个人的偏好。

<br>

> **注意:**
> 如果你在 Google Colab 上运行任何 notebook,并希望安装依赖,只需在 notebook 顶部的新单元格中运行以下代码,然后跳过本教程的其余部分:
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

下面的其余章节将介绍如何在本地机器上管理你的 Python 环境和包。

我是 [Conda](https://anaconda.org/anaconda/conda) 和 [pip](https://pypi.org/project/pip/) 的长期用户,但最近,[uv](https://github.com/astral-sh/uv) 包获得了显著的关注,因为它提供了一种更快、更高效的安装包和解析依赖的方式。

我建议从 *选项 1: 使用 uv* 开始,因为它是 2025 年更现代的方式。如果你在 *选项 1* 中遇到问题,请考虑 *选项 2: 使用 Conda*。

在本教程中,我使用的是运行 macOS 的电脑,但此工作流在 Linux 机器上类似,在其他操作系统上也可能适用。


&nbsp;
# 选项 1: 使用 uv

本节通过其 `uv pip` 接口引导你完成使用 `uv` 的 Python 设置和包安装过程。对于大多数使用过 pip 的 Python 用户来说,`uv pip` 接口可能比原生 `uv` 命令更熟悉。

&nbsp;
> **注意:**
> 还有其他安装 Python 和使用 `uv` 的方式。例如,你可以直接通过 `uv` 安装 Python,并使用 `uv add` 而不是 `uv pip install`,以获得更快的包管理体验。
>
> 如果你是 macOS 或 Linux 用户,并更喜欢原生 `uv` 命令,请参考 [./native-uv.md 教程](./native-uv.md)。我也推荐查阅官方 [`uv` 文档](https://docs.astral.sh/uv/)。
>
> `uv add` 语法同样适用于 Windows 用户。但是,我发现 `pyproject.toml` 中的某些依赖在 Windows 上会引起问题。所以,对于 Windows 用户,我推荐使用 `pixi`,它具有类似 `uv add` 的 `pixi add` 工作流。更多信息,请参阅 [./native-pixi.md 教程](./native-pixi.md)。
>
> 虽然 `uv add` 和 `pixi add` 提供了额外的速度优势,但我认为 `uv pip` 稍微更友好一些,是初学者的良好起点。但是,如果你不熟悉 Python 包管理,原生 `uv` 接口也是一个很好的机会让你从一开始就学会它。这也是我现在使用 `uv` 的方式,但我意识到如果你来自 `pip` 和 `conda`,入门门槛会更高一些。




&nbsp;
## 1. 安装 Python(如果尚未安装)

如果你以前没有手动在系统上安装过 Python,我强烈建议这样做。这有助于防止与操作系统内置 Python 安装之间可能出现的冲突,从而导致问题。

但是,即使你以前在系统上安装过 Python,也请检查你是否安装了现代版本的 Python(我推荐 3.10 或更新版本),方法是在终端执行以下命令:

```bash
python --version
```
如果它返回 3.10 或更新版本,则无需进一步操作。

&nbsp;
> **注意:**
> 如果 `python --version` 显示没有安装 Python 版本,你可能还需要检查 `python3 --version`,因为你的系统可能配置为使用 `python3` 命令。

&nbsp;
> **注意:**
> 我建议安装比最新版本至少落后 2 个版本的 Python 版本,以确保 PyTorch 兼容性。例如,如果最新版本是 Python 3.13,我建议安装 3.10 或 3.11 版本。

否则,如果未安装 Python 或版本较旧,你可以按照下面的说明为你的操作系统安装它。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-not-found.png" width="500" height="auto" alt="未找到 Python">

<br>

**Linux (Ubuntu/Debian)**

```bash
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev
```

<br>

**macOS**

如果你使用 Homebrew,用以下命令安装 Python:

```bash
brew install python@3.10
```

或者,从官方网站下载并运行安装程序:[https://www.python.org/downloads/](https://www.python.org/downloads/)。


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/python-version.png" width="700" height="auto" alt="Python 版本">

<br>

**Windows**

从官方网站下载并运行安装程序:[https://www.python.org/downloads/](https://www.python.org/downloads/)。


&nbsp;

## 2. 创建虚拟环境

我强烈建议在单独的虚拟环境中安装 Python 包,以避免修改你的操作系统可能依赖的系统级包。要在当前文件夹中创建虚拟环境,请按以下三个步骤进行。

<br>

**1. 安装 uv**

```bash
pip install uv
```

<br>

**2. 创建虚拟环境**

```bash
uv venv --python=python3.10
```

<br>

**3. 激活虚拟环境**

```bash
source .venv/bin/activate
```

&nbsp;
> **注意:**
> 如果你使用的是 Windows,你可能需要将上面的命令替换为 `source .venv/Scripts/activate` 或 `.venv/Scripts/activate`。




请注意,每次启动新的终端会话时,你都需要激活虚拟环境。例如,如果你重启了终端或电脑,第二天想继续处理该项目,只需在项目文件夹中运行 `source .venv/bin/activate` 来重新激活你的虚拟环境即可。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-1.png" width="600" height="auto" alt="虚拟环境已激活">

你也可以通过执行 `deactivate` 命令来停用该环境。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/venv-activate-2.png" width="800" height="auto" alt="虚拟环境已停用">

&nbsp;
## 3. 安装包

激活虚拟环境后,你可以使用 `uv` 安装 Python 包。例如:

```bash
uv pip install packaging
```

要从 `requirements.txt` 文件(例如位于本 GitHub 仓库顶层的那一个)安装所有必需的包,请运行以下命令,假设该文件位于你的终端会话所在的目录中:

```bash
uv pip install -r requirements.txt
```


或者,直接从仓库安装最新的依赖:

```bash
uv pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt
```


<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/uv-install.png" width="700" height="auto" alt="Uv 安装">

&nbsp;

> **注意:**
> 如果你因为某些依赖(例如,如果你使用的是 Windows)而遇到问题,你可以随时退回到使用常规 pip:
> `pip install -r requirements.txt`
> 或
> `pip install -U -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`

&nbsp;

> **Bonus 内容的可选依赖:**
> 要包含整个 bonus 内容中使用的可选依赖,请从项目根目录安装 `bonus` 依赖组:
>  `uv pip install --group bonus`
> 这在你之后查看可选的 bonus 内容时不想单独安装它们时很有用。

<br>

**完成设置**

就这样!你的环境现在应该已经准备好,可以运行本仓库中的代码了。

可选地,你可以通过执行本仓库中的 `python_environment_check.py` 脚本来运行环境检查:

```bash
python setup/02_installing-python-libraries/python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/env-check.png" width="700" height="auto" alt="环境检查">

如果你在特定包上遇到任何问题,请尝试使用以下命令重新安装它们:

```bash
uv pip install packagename
```

(这里的 `packagename` 是一个占位符名称,需要替换为你遇到问题的包的名称。)

如果问题仍然存在,请考虑在 GitHub 上 [提交一个讨论](https://github.com/rasbt/LLMs-from-scratch/discussions),或者浏览下面的 *选项 2: 使用 Conda* 章节。

<br>

**开始使用代码**

一旦一切都设置好,你就可以开始处理代码文件了。例如,通过运行以下命令启动 [JupyterLab](https://jupyterlab.readthedocs.io/en/latest/):

```bash
jupyter lab
```

&nbsp;
> **注意:**
> 如果你在使用 jupyter lab 命令时遇到问题,你也可以使用虚拟环境中的完整路径来启动它。例如,在 Linux/macOS 上使用 `.venv/bin/jupyter lab`,或在 Windows 上使用 `.venv\Scripts\jupyter-lab`。

&nbsp;

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/uv-setup/jupyter.png" width="900" height="auto" alt="Uv 安装">

&nbsp;
<br>
<br>
&nbsp;

# 选项 2: 使用 Conda



本节引导你完成通过 [miniforge](https://github.com/conda-forge/miniforge) 使用 [`conda`](https://www.google.com/search?client=safari&rls=en&q=conda&ie=UTF-8&oe=UTF-8) 的 Python 设置和包安装过程。

在本教程中,我使用的是运行 macOS 的电脑,但此工作流在 Linux 机器上类似,在其他操作系统上也可能适用。


&nbsp;
## 1. 下载并安装 Miniforge

从 GitHub 仓库 [这里](https://github.com/conda-forge/miniforge) 下载 miniforge。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/download.png" alt="download" width="600px">

根据你的操作系统,这将下载 `.sh`(macOS、Linux)或 `.exe` 文件(Windows)。

对于 `.sh` 文件,打开你的命令行终端并执行以下命令

```bash
sh ~/Desktop/Miniforge3-MacOSX-arm64.sh
```

其中 `Desktop/` 是 Miniforge 安装程序下载到的文件夹。在你的电脑上,你可能需要将其替换为 `Downloads/`。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/miniforge-install.png" alt="miniforge-install" width="600px">

接下来,逐步完成下载说明,用 "Enter" 确认。


&nbsp;
## 2. 创建新的虚拟环境

安装成功完成后,我建议创建一个名为 `LLMs` 的新虚拟环境,你可以通过执行以下命令来完成

```bash
conda create -n LLMs python=3.10
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/new-env.png" alt="new-env" width="600px">

> 许多科学计算库不会立即支持最新版本的 Python。因此,在安装 PyTorch 时,建议使用比最新版本落后一两个版本的 Python。例如,如果最新版本的 Python 是 3.13,推荐使用 Python 3.10 或 3.11。

接下来,激活你的新虚拟环境(每次打开新的终端窗口或标签页时都必须执行此操作):

```bash
conda activate LLMs
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/activate-env.png" alt="activate-env" width="600px">


&nbsp;
## 可选: 定制你的终端

如果你想像我一样美化你的终端,以便查看当前激活的是哪个虚拟环境,请查看 [Oh My Zsh](https://github.com/ohmyzsh/ohmyzsh) 项目。

&nbsp;
## 3. 安装新的 Python 库



要安装新的 Python 库,你现在可以使用 `conda` 包安装程序。例如,你可以按以下方式安装 [JupyterLab](https://jupyter.org/install) 和 [watermark](https://github.com/rasbt/watermark):

```bash
conda install jupyterlab watermark
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/conda-install.png" alt="conda-install" width="600px">



你也可以继续使用 `pip` 来安装库。默认情况下,`pip` 应该链接到你的新 `LLms` conda 环境:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/check-pip.png" alt="check-pip" width="600px">

&nbsp;
## 4. 安装 PyTorch

PyTorch 可以像其他任何 Python 库或包一样使用 pip 进行安装。例如:

```bash
pip install torch
```

但是,由于 PyTorch 是一个功能全面的库,同时支持 CPU 和 GPU 代码,其安装可能需要额外的设置和说明(更多信息请参阅书中的 *A.1.3 安装 PyTorch*)。

还强烈建议查阅 PyTorch 官方网站的安装指南菜单 [https://pytorch.org](https://pytorch.org)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/01_optional-python-setup-preferences/pytorch-installer.jpg" width="600px">

&nbsp;
## 5. 安装本书中使用的 Python 包和库

请参考 [安装本书中使用的 Python 包和库](../02_installing-python-libraries/README.md) 文档,了解如何安装所需库。

<br>

---




有任何问题吗?请随时在 [讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 中与我们联系。