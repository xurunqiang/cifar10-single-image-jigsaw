# v3：Tiny ImageNet 拼图与整图表征

## 默认方案

- 官方训练集按类别拆成 90,000 张训练、10,000 张开发集，seed=42；官方验证集用于最终评估。
- 64×64 图片右侧、底部复制填充到 65×65，切成 5×5 个 13×13 碎片。中心真实块作为起点，按上、右、下、左的 BFS 顺序填入其余 24 块，不重复选块。
- 种子不计入预测损失和块准确率；作为上下文时允许梯度经过共享编码器。
- CNN 通道 64/128/256，每个阶段两个残差块；局部分支 2 层、全局分支 4 层，均为 8 头，特征维度 256。
- 无坐标虚拟 token 通过交叉注意力读取自主拼好的整图 Hcand，再经过 MLP 得到 Qvirtual；与 Hcand 做余弦匹配和 softmax 加权汇聚，得到 Zvirtual。真实块保留预测位置，虚拟 token 没有自身物理位置。
- 投影头为 256→512→512，中间 LayerNorm、GELU；两次整图增强的投影输出计算 VICReg。统计计算显式关闭 autocast，使用 FP32。
- 100 轮训练：TF 在 1–20 轮为 1，21–80 轮降到 0，81–100 轮自主拼图；语义损失权重在前 10 轮从 0.01 升到 0.1。更改总轮数时 TF 阶段按比例调整。
- 语义分支始终使用自主预测的棋盘；TF 阶段额外的离散拼图过程不求梯度，并复用 CNN 特征。随后用原始可求导特征重新计算整图表征。

## 训练和续训

在项目根目录执行：

```bash
python -m v3.run_experiments train --save_dir checkpoints/v3_joint
python -m v3.run_experiments train --resume checkpoints/v3_joint/latest.pt
```

保存 latest.pt、best_jigsaw.pt、best_feat.pt、history.json 和 training_curves.png。best_jigsaw 按开发集非种子块准确率选择，同分比较整图准确率。

默认每 5 轮和最后一轮，使用均衡抽取的 10,000 张训练参考图和 10,000 张开发图评估 Zvirtual 的 5-NN，选择 best_feat。该步骤需要额外推理时间。可用 `--feature_eval_interval` 调整频率；设为 0 则不评估、不生成 best_feat。样本数由 `--feature_reference_samples` 和 `--feature_validation_samples` 控制。

类别标签不进入拼图或 VICReg 损失，但 **best_feat 的选择使用开发集类别标签**，不能称为完全不使用标签的模型选择。

纯拼图对照：

```bash
python -m v3.run_experiments train --semantic_weight 0 --save_dir checkpoints/v3_jigsaw
```

纯拼图对照的虚拟模块没有语义损失训练，应同时比较 raw_mean 和 hcand_mean。支持 `--grid_size 3/5/7`，中心和填充自动计算；主实验仍为 5×5。

## 最终评估及效果图

```bash
python -m v3.evaluate --checkpoint checkpoints/v3_joint/best_feat.pt --eval_repr --features_npz visualizations/v3/features.npz --visualization_dir visualizations/v3/retrieval --output_json visualizations/v3/metrics.json
python -m v3.run_experiments visualize --checkpoint checkpoints/v3_joint/best_jigsaw.pt --output_dir visualizations/v3/random
python -m v3.run_experiments visualize --checkpoint checkpoints/v3_joint/best_jigsaw.pt --failures --output_dir visualizations/v3/failures
```

默认最终评估集为官方验证集。特征评估比较 raw_mean、hcand_mean、z_virtual，提供 5-NN、K-means 准确率、ARI、NMI。默认训练参考图和评估图各最多 10,000 张，按类别均衡抽样；可调整 reference_samples、representation_samples。

K-means 对 L2 归一化的训练参考特征拟合 200 个中心，拟合不使用标签。之后用训练参考标签确定 Hungarian 类别映射，再冻结中心和映射评估新图。评估集标签只评分，不拟合中心或类别映射；该准确率与旧版在评估集上聚类、匹配的分数不宜直接比较。

随机拼图图和失败图分开输出，附带抽样记录；检索图对三种特征使用相同随机查询图。增加 `--random_init --random_seed 3407` 可评估相同结构的随机初始化基线。

## 检查点兼容性与检查范围

本次修正改变了虚拟查询和投影头结构，旧版 v3 检查点不能直接续训。新检查点使用 format_version=2，并记录模型、训练、数据增强配置、划分指纹及随机状态；续训时检查一致性。已有文件没有迁移或覆盖。

本次按要求只进行代码静态检查和修改，没有重跑训练、推理或测试。显存占用、速度和收敛效果仍需实际运行确认。
