# 故障排查指南

本页汇总了阅读本书时遇到的常见问题与配置提示。

&nbsp;
## Notebook 图片加载问题

各章 Notebook 使用托管在 `https://sebastianraschka.com/images/LLMs-from-scratch-images/...` 的 Markdown 图片链接。这种做法能让仓库的下载体积保持可控，但也意味着图片的可用性依赖于图床服务器与你的网络连接。

如果 `.ipynb` Notebook 中的图片无法渲染：

- 在浏览器中直接打开某个图片 URL，例如 [https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp](https://sebastianraschka.com/images/LLMs-from-scratch-images/ch02_compressed/02.webp)。
- 如果浏览器中同样无法加载该 URL，那么问题大概率是网站临时不可用、DNS、VPN、代理、防火墙或本地网络问题，而不是 Notebook 本身的问题。
- 建议在不同设备或网络上重复验证该 URL（例如尝试用手机打开同一张图片）；如果手机上能正常加载，说明很可能是你电脑上的 VPN 或防火墙问题。
- 如果手机上也加载不出来，欢迎提交 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues) 帮助我进一步排查。

&nbsp;
## 在更新仓库的同时保留你自己的 Notebook 修改

如果你希望在修改 Notebook 的同时仍然能接收到仓库更新，请先 fork 本仓库，再克隆 fork。主仓库的 Notebook 与印刷书保持一致，通常不会变动，只在关键修复时例外。多数仓库更新只新增附加材料。

Notebook 文件本质上是 JSON，因此 Git diff 与合并冲突会比较难读。为了避免无谓的冲突，建议把你的实验与受版本控制的书中 Notebook 分开：

- 在修改前先复制一份 Notebook，例如从 `ch02.ipynb` 复制为 `ch02_experiments.ipynb`。
- 把你的草稿 Notebook 放在单独的文件夹里，或放在你自己的分支上。
- 通过 `upstream` 远端从原仓库拉取更新，仅在需要这些更新时再 merge 或 rebase。

创建 fork 并克隆它：

1. 打开 [https://github.com/rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
2. 在 GitHub 右上角点击 **Fork** 按钮。
3. 克隆你的 fork，把 `YOUR-USERNAME` 替换为你的 GitHub 用户名：

```bash
git clone https://github.com/YOUR-USERNAME/LLMs-from-scratch.git
cd LLMs-from-scratch
```

接着把原仓库添加为 `upstream`，方便后续拉取更新：

```bash
git remote add upstream https://github.com/rasbt/LLMs-from-scratch.git
git fetch upstream
git merge upstream/main
```

如果确实需要合并已编辑的 Notebook，可以考虑安装 [`nbdime`](https://nbdime.readthedocs.io/) 以获得针对 Notebook 的 diff 与合并工具：

```bash
pip install nbdime
nbdime config-git --enable
```

更多上下文参见 [#1015](https://github.com/rasbt/LLMs-from-scratch/issues/1015)。

&nbsp;
## Apple Silicon 与 MPS 支持

部分 Notebook 和脚本在有 `cuda` 时使用 `cuda`，否则回退到 `cpu`，并未选择 Apple 的 `mps` 后端。在很多场景下故意省略 `mps`，是因为早期 PyTorch/MPS 版本在多个例子里（尤其是训练和微调）会产生不稳定或不一致的结果。

如果你在使用 Apple Silicon Mac，并且观察到损失曲线发散、损失出现陡峭尖峰、生成文本质量差，或者结果与书里不吻合，请先用 `cpu` 重跑这个例子。为了获得更快的训练速度并与书中行为一致，建议在本地 NVIDIA GPU 或云 GPU 上使用 `cuda`。

较新版本的 PyTorch 可能改善了 MPS 行为，如果你能仔细验证结果，可以在本地尝试 `mps`。但如果你自己给脚本加入 `mps` 支持，请注意 `pin_memory=True`、`torch.compile` 以及 DDP/多 GPU 代码等 CUDA-specific 选项可能需要单独加保护。

更多上下文参见 [#977](https://github.com/rasbt/LLMs-from-scratch/issues/977)、[#625](https://github.com/rasbt/LLMs-from-scratch/discussions/625)、[#644](https://github.com/rasbt/LLMs-from-scratch/discussions/644)、[#442](https://github.com/rasbt/LLMs-from-scratch/discussions/442) 和 [#846](https://github.com/rasbt/LLMs-from-scratch/issues/846)。

&nbsp;
## 其他问题

如果遇到其他问题，欢迎新建一个 GitHub [Issue](https://github.com/rasbt/LLMs-from-scratch/issues)。