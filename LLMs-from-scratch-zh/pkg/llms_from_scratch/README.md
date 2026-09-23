# `llms-from-scratch` PyPI 包

这个可选的 PyPI 包让你可以方便地导入《*Build a Large Language Model From Scratch*（从零构建大语言模型）》一书中各章节的代码。

&nbsp;
## 安装

&nbsp;
### 从 PyPI 安装

从官方 [Python Package Index](https://pypi.org/project/llms-from-scratch/) (PyPI) 安装 `llms-from-scratch` 包：

```bash
pip install llms-from-scratch
```

> **注意：** 如果你使用 [`uv`](https://github.com/astral-sh/uv)，把 `pip` 换成 `uv pip`，或者使用 `uv add`：

```bash
uv add llms-from-scratch
```



&nbsp;
### 从 GitHub 以可编辑方式安装

如果你想修改代码并希望改动在开发过程中实时生效：

```bash
git clone https://github.com/rasbt/LLMs-from-scratch.git
cd LLMs-from-scratch
pip install -e .
```

> **注意：** 使用 `uv` 时，可改为：

```bash
uv add --editable . --dev
```



&nbsp;
## 使用该包

安装完成后，你可以通过以下方式导入任意章节的代码：

```python
from llms_from_scratch.ch02 import GPTDatasetV1, create_dataloader_v1

from llms_from_scratch.ch03 import (
    SelfAttention_v1,
    SelfAttention_v2,
    CausalAttention,
    MultiHeadAttentionWrapper,
    MultiHeadAttention,
    PyTorchMultiHeadAttention # 附加：使用 PyTorch 的 scaled_dot_product_attention 实现的更高效版本
)

from llms_from_scratch.ch04 import (
    LayerNorm,
    GELU,
    FeedForward,
    TransformerBlock,
    GPTModel,
    GPTModelFast # 附加：使用 PyTorch 的 scaled_dot_product_attention 实现的更高效版本
    generate_text_simple
)

from llms_from_scratch.ch05 import (
    generate,
    train_model_simple,
    evaluate_model,
    generate_and_print_sample,
    assign,
    load_weights_into_gpt,
    text_to_token_ids,
    token_ids_to_text,
    calc_loss_batch,
    calc_loss_loader,
    plot_losses,
    download_and_load_gpt2
)

from llms_from_scratch.ch06 import (
    download_and_unzip_spam_data,
    create_balanced_dataset,
    random_split,
    SpamDataset,
    calc_accuracy_loader,
    evaluate_model,
    train_classifier_simple,
    plot_values,
    classify_review
)

from llms_from_scratch.ch07 import (
    download_and_load_file,
    format_input,
    InstructionDataset,
    custom_collate_fn,
    check_if_running,
    query_model,
    generate_model_scores
)


from llms_from_scratch.appendix_a import NeuralNetwork, ToyDataset

from llms_from_scratch.appendix_d import find_highest_gradient, train_model
```



&nbsp;

### GPT-2 KV cache 变体（附加材料）

```python
from llms_from_scratch.kv_cache.gpt2 import GPTModel
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

关于 KV caching 的更多信息，请参阅 [KV cache README](../../ch04/03_kv-cache)。



&nbsp;

### Llama 3（附加材料）

```python
from llms_from_scratch.llama3 import (
		load_weights_into_llama,
  	Llama3Model,
    Llama3ModelFast,
    Llama3Tokenizer,
    ChatFormat,
    clean_text
)

# KV cache 即插即用替代
from llms_from_scratch.kv_cache.llama3 import Llama3Model
from llms_from_scratch.kv_cache.generate import generate_text_simple
```

关于 `llms_from_scratch.llama3` 的使用信息，请参见[该附加章节](../../ch05/07_gpt_to_llama/README.md)。

关于 KV caching 的更多信息，请参阅 [KV cache README](../../ch04/03_kv-cache)。


&nbsp;
### Qwen3（附加材料）

```python
from llms_from_scratch.qwen3 import (
    load_weights_into_qwen,
    Qwen3Model,
    Qwen3Tokenizer,
)

# KV cache 即插即用替代
from llms_from_scratch.kv_cache.qwen3 import Qwen3Model
from llms_from_scratch.kv_cache.generate import (
    generate_text_simple,
    generate_text_simple_stream
)

# 支持批量推理的 KV cache 即插即用替代
from llms_from_scratch.kv_cache_batched.generate import (
    generate_text_simple,
    generate_text_simple_stream
)
from llms_from_scratch.kv_cache_batched.qwen3 import Qwen3Model
```

关于 `llms_from_scratch.qwen3` 的使用信息，请参见[该附加章节](../../ch05/11_qwen3/README.md)。

关于 KV caching 的更多信息，请参阅 [KV cache README](../../ch04/03_kv-cache)。