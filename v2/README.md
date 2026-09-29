# v2 单图拼图

一个已知真实位置的种子，加上严格不重复的逐块选取。种子不计位置损失和 Patch 准确率；其内容特征仍通过其他块的损失学习。局部四方向注意力和全局碎片集合注意力共同为每一步打分。

## 训练与评估约定

- CIFAR-10 官方训练集固定划分 45,000 / 5,000 张；官方测试集只用于最终评估。划分由 `--seed` 决定。
- 训练的种子、碎片排列和随机边界顺序随样本与 epoch 改变；验证的这三项固定，不受批次大小或训练随机状态影响。
- 默认 100 轮：前 20 轮正确上下文，中间 60 轮提示概率线性降至零，最后 20 轮自主拼接。其他轮数按比例分配；1 轮仅正确上下文，2 轮分别为正确上下文和自主拼接。
- 分类损失在所有非种子候选上计算，实际放置只允许未用候选。正确块已被提前用掉时，损失仍有限，teacher forcing 自动退回预测。
- `Tr Context` 是提示覆盖前、当前混合上下文中的选块准确率；`Tr Placed` 包含正确提示。真正的自主表现看 `Val Patch/Perf`。
- `UsedGT` 统计正确候选已被提前占用的比例，独立于本步是否要求提示。返回指标还包括实际提示比例和提示失败比例。
- 相邻块准确率检查真实的有向上下/左右关系，正确块对整体平移后仍计为正确；跨原图行边界不计为相邻。
- 模型使用标准化后的 CIFAR 图像；`solve` 返回的图像保持输入数值范围。若用于展示，需要用相同 CIFAR 均值和标准差反归一化。

## 命令

在项目根目录运行。以下目录与旧检查点分开，便于重新训练比较：

```bash
python -m v2.run_experiments --grid-size 3 --epochs 100 \
  --strategy random_frontier --save-dir-prefix checkpoints/v2_reviewed

python -m v2.run_experiments \
  --resume checkpoints/v2_reviewed/grid3/latest.pt

python -m v2.run_experiments \
  --eval-checkpoint checkpoints/v2_reviewed/grid3/best.pt
```

续训使用检查点保存的模型、数据、批次、课程与优化器配置；运行设备可用 `--device cpu` 或 `--device cuda:0` 指定。检查点不存在会明确报错。

- 扩展：`--strategy bfs` 或 `random_frontier`。
- 分支对照：`--mode both`、`local_only`、`global_only`。非默认分支/策略自动追加到保存目录，避免覆盖默认实验。
- 种子：`--seed-selection random`、`center`，或 `custom --seed-coord 1 1`。
- 小规模检查：`--max-train-samples 4 --max-val-samples 3 --batch-size 2 --epochs 3`。
- 最终评估默认使用 42、101、202 三组固定配置，分别评估两种扩展策略；可用 `--max-eval-samples` 限制冒烟检查规模。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest v2/tests -q
```

## 旧检查点

修订前的 v2 使用全部官方训练集训练，并使用官方测试集选最佳模型；随机边界评估顺序也未固定。旧权重可以加载，但旧成绩不能与新的独立验证成绩直接比较。加载旧检查点会重置最佳指标并提示旧评估协议；要得到干净的验证结果，应从头训练。原有权重文件不会被审查或测试修改。

小规模训练与测试只检查实现正确性，不代表完整数据上的精度提升。
