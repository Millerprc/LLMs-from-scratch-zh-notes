# 安装本书中使用的 Python 包和库

本文档提供了有关仔细检查你安装的 Python 版本和包的更多信息。(请参阅 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 文件夹,了解有关安装 Python 和 Python 包的更多信息。)

我为本使用了 [这里](https://github.com/rasbt/LLMs-from-scratch/blob/main/requirements.txt) 列出的库。这些库的新版本也可能兼容。但是,如果你在使用代码时遇到任何问题,可以尝试使用这些库版本作为回退方案。



> **注意:**
> 如果你使用的是 [选项 1: 使用 uv](../01_optional-python-setup-preferences/README.md) 中描述的 `uv`,你可以在下面的命令中将 `pip` 替换为 `uv pip`。例如,`pip install -r requirements.txt` 变成 `uv pip install -r requirements.txt`



要最方便地安装这些依赖,你可以使用本代码仓库根目录中的 `requirements.txt` 文件,并执行以下命令:

```bash
pip install -r requirements.txt
```

或者,你也可以通过 GitHub URL 安装它,如下所示:

```bash
pip install -r https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/requirements.txt
```


然后,在完成安装后,请使用以下命令检查所有包是否都已安装并且是最新版本

```bash
python python_environment_check.py
```

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_1.jpg" width="600px">

还建议通过运行此目录中的 `python_environment_check.ipynb` 在 JupyterLab 中检查版本,理想情况下应该得到与上面相同的结果。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/check_2.jpg" width="500px">

如果你看到以下问题,很可能是你的 JupyterLab 实例连接到了错误的 conda 环境:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/jupyter-issues.jpg" width="450px">

在这种情况下,你可能想使用 `watermark` 并通过 `--conda` 标志来检查你是否在正确的 conda 环境中打开了 JupyterLab 实例:

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/watermark.jpg" width="350px">


&nbsp;
## 安装 PyTorch

PyTorch 可以像其他任何 Python 库或包一样使用 pip 进行安装。例如:

```bash
pip install torch
```

但是,由于 PyTorch 是一个功能全面的库,同时支持 CPU 和 GPU 代码,其安装可能需要额外的设置和说明(更多信息请参阅书中的 *A.1.3 安装 PyTorch*)。

还强烈建议查阅 PyTorch 官方网站的安装指南菜单 [https://pytorch.org](https://pytorch.org)。

<img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/setup/02_installing-python-libraries/pytorch-installer.jpg" width="600px">

<br>



&nbsp;
## JupyterLab 提示

如果你是在 JupyterLab 而不是 VSCode 中查看 notebook 代码,请注意 JupyterLab(在其默认设置中)在最近版本中存在滚动 bug。我的建议是进入 Settings -> Settings Editor 并将 "Windowing mode" 更改为 "none"(如下图所示),这似乎可以解决此问题。


![Jupyter 故障 1](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_1.webp)

<br>

![Jupyter 故障 2](https://sebastianraschka.com/images/reasoning-from-scratch-images/bonus/setup/jupyter_glitching_2.webp)

<br>

---




有任何问题吗?请随时在 [讨论论坛](https://github.com/rasbt/LLMs-from-scratch/discussions) 中与我们联系。