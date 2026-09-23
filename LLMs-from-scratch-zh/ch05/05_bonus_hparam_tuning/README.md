# 为预训练优化超参数

[hparam_search.py](hparam_search.py) 脚本基于 [Appendix D: Adding Bells and Whistles to the Training Loop](../../appendix-D/01_main-chapter-code/appendix-D.ipynb) 中扩展的训练函数，旨在通过网格搜索找到最优的超参数。

>[!NOTE]
此脚本的运行将需要很长时间。你可能希望减少顶部 `HPARAM_GRID` 字典中探索的超参数配置数量。