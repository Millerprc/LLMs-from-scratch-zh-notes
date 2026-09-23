# 可选的设置说明


本文档列出了多种在你的机器上设置环境并运行本仓库代码的方法。我建议你从上到下浏览各个章节,然后决定哪种方式最适合自己的需求。

&nbsp;

## 快速开始

如果你机器上已经安装了 Python,最快的开始方式是从本代码仓库的根目录执行下面的 pip 安装命令,从 [../requirements.txt](../requirements.txt) 文件安装所需的依赖:

```bash
pip install -r requirements.txt
```

<br>

> **注意:** 如果你在 Google Colab 上运行任何 notebook,并希望安装依赖,只需在 notebook 顶部的新单元格中运行以下代码:
> `pip install uv && uv pip install --system -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/refs/heads/main/requirements.txt`
> 另外,在克隆仓库后,你可以从项目根目录运行 `uv pip install --group bonus` 来一次性安装所有 bonus 内容的依赖。这样在你之后查看可选的 bonus 内容时就不必再单独安装了。

<br>
<br>

在下面的视频中,我分享了我个人在电脑上设置 Python 环境的方法:

<br>
<br>

[![视频链接](https://img.youtube.com/vi/yAcWnfsZhzo/0.jpg)](https://www.youtube.com/watch?v=yAcWnfsZhzo)


&nbsp;
# 本地设置

本节提供了在本地运行本书代码的建议。请注意,本书主要章节的代码被设计为可在普通笔记本电脑上在合理的时间内执行,不需要专门的硬件。我在 M3 MacBook Air 笔记本上测试了所有主要章节。此外,如果你的笔记本或台式机配有 NVIDIA GPU（图形处理器）,代码会自动利用它。

&nbsp;
## 设置 Python

如果你还没有在机器上安装 Python,我已经在以下目录中写下了我个人的 Python 设置偏好:

- [01_optional-python-setup-preferences](./01_optional-python-setup-preferences)
- [02_installing-python-libraries](./02_installing-python-libraries)

下面的 *使用 DevContainers（开发容器）* 部分介绍了另一种在机器上安装项目依赖的方法。

&nbsp;

## 使用 Docker DevContainers

作为上面 *设置 Python* 章节的替代方案,如果你偏好一种能隔离项目依赖和配置的开发环境,使用 Docker 是一个非常有效的解决方案。这种方法无需手动安装软件包和库,并保证了一致的开发环境。你可以在下面的链接中找到更多关于设置 Docker 和使用 DevContainer 的说明:

- [03_optional-docker-environment](03_optional-docker-environment)

&nbsp;

## Visual Studio Code 编辑器

代码编辑器有很多不错的选择。我个人偏爱的是流行的开源 [Visual Studio Code (VSCode)](https://code.visualstudio.com) 编辑器,它可以通过许多有用的插件和扩展轻松增强(更多信息请参阅下面的 *VSCode 扩展* 章节)。在 [VSCode 官网](https://code.visualstudio.com) 上可以找到 macOS、Linux 和 Windows 的下载说明。

&nbsp;

## VSCode 扩展

如果你使用 Visual Studio Code (VSCode) 作为主要的代码编辑器,你可以在 `.vscode` 子文件夹中找到推荐的扩展。这些扩展提供增强的功能和对本仓库有用的工具。

要安装它们,请在 VSCode 中打开此 "setup" 文件夹(File -> Open Folder...),然后点击右下角弹出菜单中的 "Install" 按钮。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/vs-code-extensions.webp?1" alt="1" width="700">

或者,你也可以将 `.vscode` 扩展文件夹移动到这个 GitHub 仓库的根目录:

```bash
mv setup/.vscode ./
```

之后,VSCode 会在每次打开 `LLMs-from-scratch` 主文件夹时自动检查你系统上是否已经安装了推荐的扩展。

&nbsp;

# 云端资源

本节介绍用于运行本书代码的云端替代方案。

虽然代码可以在没有专用 GPU 的普通笔记本和台式机上运行,但配有 NVIDIA GPU 的云平台可以显著提升代码的运行时间,特别是第 5 至 7 章。

&nbsp;

## 使用 Lightning Studio

为了在云端获得流畅的开发体验,我推荐 [Lightning AI Studio](https://lightning.ai/) 平台,它允许用户在云端 CPU 和 GPU 上设置持久化环境,并同时使用 VSCode 和 Jupyter Lab。

启动一个新的 Studio 后,你可以打开终端,执行以下设置步骤来克隆仓库并安装依赖:

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -r requirements.txt
```

(与 Google Colab 不同的是,这些步骤只需执行一次,因为 Lightning AI Studio 环境是持久化的,即使你在 CPU 和 GPU 机器之间切换。)

然后,导航到你想运行的 Python 脚本或 Jupyter Notebook。你也可以选择性地连接一个 GPU 来加速代码的运行时间,例如,在第 5 章预训练 LLM,或在第 6、7 章对它进行微调时。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/studio.webp" alt="1" width="700">

&nbsp;

## 使用 Google Colab

要在云端使用 Google Colab 环境,请前往 [https://colab.research.google.com/](https://colab.research.google.com/),从 GitHub 菜单打开对应的章节 notebook,或者如下图所示将 notebook 拖入 *Upload* 区域。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_1.webp" alt="1" width="700">


同时请确保将相关文件(数据集文件和 notebook 所导入的 .py 文件)也上传到 Colab 环境,如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_2.webp" alt="2" width="700">


你也可以通过更改 *Runtime* 让代码运行在 GPU 上,如下图所示。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/README/colab_3.webp" alt="3" width="700">


&nbsp;

# 有疑问?

如果你有任何问题,请随时通过本 GitHub 仓库的 [Discussions](https://github.com/rasbt/LLMs-from-scratch/discussions) 论坛与我们联系。