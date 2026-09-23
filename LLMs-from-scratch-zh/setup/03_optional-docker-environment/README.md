# Docker 环境设置指南

如果你偏好一种能隔离项目依赖和配置的开发环境,使用 Docker 是一个非常有效的解决方案。这种方法无需手动安装软件包和库,并保证了一致的开发环境。

如果你相对于 [../01_optional-python-setup-preferences](../01_optional-python-setup-preferences) 和 [../02_installing-python-libraries](../02_installing-python-libraries) 中解释的 conda 方式,更倾向于使用 Docker,本指南将引导你完成为本书设置可选 Docker 环境的过程。

<br>

## 下载并安装 Docker

开始使用 Docker 最简单的方法是为你的相关平台安装 [Docker Desktop](https://docs.docker.com/desktop/)。

Linux (Ubuntu) 用户可能更愿意安装 [Docker Engine](https://docs.docker.com/engine/install/ubuntu/),并按照 [安装后](https://docs.docker.com/engine/install/linux-postinstall/) 步骤操作。

<br>

## 在 Visual Studio Code 中使用 Docker DevContainer

Docker DevContainer(Development Container,开发容器)是一个允许开发者将 Docker 容器用作成熟开发环境的工具。这种方法确保用户可以快速启动并运行一致的开发环境,无论他们的本地机器设置如何。

虽然 DevContainers 也能与其他 IDE 配合使用,但与 DevContainers 一起使用的常用 IDE/编辑器是 Visual Studio Code (VS Code)。下面的指南解释了如何在 VS Code 上下文中使用本书的 DevContainer,但类似的过程也应适用于 PyCharm。如果尚未安装并且想使用它,请 [安装](https://code.visualstudio.com/download)。

1. 克隆此 GitHub 仓库,并 `cd` 进入项目根目录。

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
```

2. 将 `.devcontainer` 文件夹从 `setup/03_optional-docker-environment/` 移动到当前目录(项目根目录)。

```bash
mv setup/03_optional-docker-environment/.devcontainer ./
```

3. 在 Docker Desktop 中,确保 **_desktop-linux_ builder** 正在运行并将被用于构建 Docker 容器(请参阅 _Docker Desktop_ -> _Change settings_ -> _Builders_ -> _desktop-linux_ -> _..._ -> _Use_)

4. 如果你有 [CUDA-supported GPU（支持 CUDA 的图形处理器）](https://developer.nvidia.com/cuda-gpus),你可以加速训练和推理:

    4.1 按照 [此处](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#installing-with-apt) 的说明安装 **NVIDIA Container Toolkit**。NVIDIA Container Toolkit 支持情况如 [此处](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#nvidia-compute-software-support-on-wsl-2) 所述。

    4.2 在 Docker Engine 守护进程配置中将 _nvidia_ 添加为运行时(请参阅 _Docker Desktop_ -> _Change settings_ -> _Docker Engine_)。将这些行添加到你的配置:

```json
"runtimes": {
    "nvidia": {
    "path": "nvidia-container-runtime",
    "runtimeArgs": []
}
```

例如,完整的 Docker Engine 守护进程配置 json 代码应如下所示:

```json
{
  "builder": {
    "gc": {
      "defaultKeepStorage": "20GB",
      "enabled": true
    }
  },
  "experimental": false,
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
```

并重启 Docker Desktop。

5. 在终端中输入 `code .` 以在 VS Code 中打开项目。或者,你可以启动 VS Code 并从 UI 中选择要打开的项目。

6. 从 VS Code 左侧的 _Extensions_ 菜单安装 **Remote Development** 扩展。

7. 打开 DevContainer。

由于 `.devcontainer` 文件夹存在于主 `LLMs-from-scratch` 目录中(以 `.` 开头的文件夹在你的操作系统中可能是不可见的,具体取决于你的设置),VS Code 应自动检测到它,并询问你是否想在 devcontainer 中打开项目。如果没有,只需按 `Ctrl + Shift + P` 打开命令面板,然后开始输入 `dev containers`,即可看到所有 DevContainer 特定选项的列表。



&nbsp;
> ⚠️ **关于以 root 身份运行的说明**
>
> 默认情况下,DevContainer 以 *root 用户* 身份运行。出于安全原因,通常不建议这样做,但为了本书设置的简洁性,使用了 root 配置,以便所有必需的包都能干净地安装在容器内。
>
> 如果你尝试在容器内手动启动 Jupyter Lab,你可能会看到此错误:
>
>   ```bash
>   Running as root is not recommended. Use --allow-root to bypass.
>   ```
>
>   在这种情况下,你可以运行:
>
>   ```bash
>   uv run jupyter lab --allow-root
>   ```
>
>   - 当将 VS Code 与 Jupyter 扩展一起使用时,通常不需要手动启动 Jupyter Lab。通过扩展打开 notebook 应该可以直接使用。
>   - 偏好更严格安全性的高级用户可以修改 `.devcontainer.json` 来设置非 root 用户,但这需要额外的配置,对于大多数用例来说不是必需的。



8. 选择 **Reopen in Container**。

如果之前没有构建过 `.devcontainer` 配置中指定的 Docker 镜像,Docker 现在将开始构建过程,或者如果可以从注册表获取该镜像则拉取它。

整个过程是自动化的,可能需要几分钟时间,具体取决于你的系统和网速。可选地,点击 VS Code 右下角的 "Starting Dev Container (show log)" 以查看当前构建进度。

完成后,VS Code 将自动连接到容器,并在新创建的 Docker 开发环境中重新打开项目。你将能够像在本地机器上运行一样编写、执行和调试代码,但额外具有 Docker 的隔离性和一致性优势。

&nbsp;
> **警告:**
> 如果你在构建过程中遇到错误,这很可能是因为你的机器不支持 NVIDIA container toolkit,因为你的机器没有兼容的 GPU。在这种情况下,请编辑 `devcontainer.json` 文件以删除 `"runArgs": ["--runtime=nvidia", "--gpus=all"],` 行,然后再次运行 "Reopen Dev Container" 过程。

9. 完成。

一旦镜像被拉取和构建完成,你的项目应该已挂载在容器内,并已安装所有包,可以进行开发了。

<br>

## 卸载 Docker 镜像

以下是如果你不再计划使用 DevContainer,卸载或删除 Docker 容器和镜像的说明。此过程不会从你的系统中删除 Docker 本身,而是清理特定于项目的 Docker 构件。

1. 列出所有 Docker 镜像,以查找与你的 DevContainer 关联的镜像:

```bash
docker image ls
```

2. 使用其镜像 ID 或名称删除 Docker 镜像:

```bash
docker image rm [IMAGE_ID_OR_NAME]
```

<br>

## 卸载 Docker

如果你决定 Docker 不适合你,并希望卸载它,请参阅 [此处](https://docs.docker.com/desktop/uninstall/) 的官方文档,其中概述了针对你特定操作系统的步骤。