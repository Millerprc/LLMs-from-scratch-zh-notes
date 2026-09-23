# 为训练循环添加各种改进

主章节使用了一个相对简单的训练函数，以保持代码的可读性并使第 5 章适合页数限制。可以选择性地，我们添加线性 warmup、cosine decay 调度以及梯度裁剪，以改善训练的稳定性和收敛性。

你可以在 [Appendix D: Adding Bells and Whistles to the Training Loop](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中找到这个更复杂的训练函数的代码。