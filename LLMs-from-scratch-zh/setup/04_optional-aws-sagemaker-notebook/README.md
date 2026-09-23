# AWS CloudFormation 模板: 带有 LLMs-from-scratch 仓库的 Jupyter Notebook

此 CloudFormation 模板在 Amazon SageMaker 中创建了一个支持 GPU 的 Jupyter notebook,包含一个执行角色和 LLMs-from-scratch GitHub 仓库。

## 它能做什么:

1. 创建一个 IAM 角色,具有 SageMaker notebook 实例所需的权限。
2. 创建一个 KMS 密钥和别名,用于加密 notebook 实例。
3. 配置 notebook 实例生命周期配置脚本,该脚本:
   - 在用户的主目录中安装单独的 Miniconda 安装。
   - 创建一个自定义 Python 环境,包含 TensorFlow 2.15.0 和 PyTorch 2.1.0,两者均支持 CUDA。
   - 安装其他有用的包,如 Jupyter Lab、Matplotlib 等。
   - 将自定义环境注册为 Jupyter kernel。
4. 使用指定的配置创建 SageMaker notebook 实例,包括支持 GPU 的实例类型、执行角色和默认代码仓库。

## 如何使用:

1. 下载 CloudFormation 模板文件 (`cloudformation-template.yml`)。
2. 在 AWS Management Console 中,导航到 CloudFormation 服务。
3. 创建一个新的 stack,并上传模板文件。
4. 为 notebook 实例提供一个名称(例如,"LLMsFromScratchNotebook")(默认为 LLMs-from-scratch GitHub 仓库)。
5. 检查并接受模板的参数,然后创建 stack。
6. 一旦 stack 创建完成,SageMaker notebook 实例将在 SageMaker 控制台中可用。
7. 打开 notebook 实例,并开始使用预配置的环境来处理你的 LLMs-from-scratch 项目。

## 关键要点:

- 模板创建了一个支持 GPU 的 (`ml.g4dn.xlarge`) notebook 实例,具有 50GB 存储空间。
- 它设置了一个自定义 Miniconda 环境,包含 TensorFlow 2.15.0 和 PyTorch 2.1.0,两者均支持 CUDA。
- 自定义环境被注册为 Jupyter kernel,可在 notebook 中使用。
- 该模板还会创建一个 KMS 密钥用于加密 notebook 实例,以及一个具有必要权限的 IAM 角色。