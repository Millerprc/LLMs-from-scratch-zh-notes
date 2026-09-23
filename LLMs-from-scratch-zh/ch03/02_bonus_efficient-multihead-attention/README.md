# 更高效的多头注意力实现（More Efficient Multi-Head Attention Implementations）

- [mha-implementations.ipynb](mha-implementations.ipynb) 包含多头注意力（multi-head attention）的不同实现，并进行对比



下面这些图总结了性能基准（越低越好）。



#### 仅前向（Forward pass only）

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/1_forward-only.webp?1" width="500px"></a>



#### 前向 + 反向（Forward and backward pass）

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/2_forward-and-backward.webp?1" width="500px"></a>



#### 编译后的前向 + 反向（Forward and backward pass after compilation）

<a href="mha-implementations.ipynb"><img src="https://sebastianraschka.com/images/LLMs-from-scratch-images/bonus/mha-benchmark/3_forward-and-backward-compiled.webp?1" width="500px"></a>