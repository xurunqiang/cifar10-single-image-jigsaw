# 单图拼图还原系统 (v1 版本)

基于 CIFAR-10 的单图打乱、真实物理位置种子锚定、BFS 队列向外生长扩展、局部十字注意力与全局 Transformer 融合的拼图复原系统。

## 核心设计与特性

1. **单图闭环打乱**：
   - 图像切分成 $3\times3$（9块）、$5\times5$（25块）或 $7\times7$（49块）Patch。
   - 仅在单图内部打乱，候选池规模与网格块数严格一致，彻底剔除跨图大池。
2. **真实物理位置锚定**：
   - 随机选取的种子块放置在画布对应的**真实物理网格坐标** $(r_{\text{seed}}, c_{\text{seed}})$ 上，以此作为 BFS 队列原点向四周扩散。
3. **入队出队向外补全（BFS 队列生长）**：
   - 从种子出发，依次扫描四周正交相邻（上、右、下、左）未填补槽位。
   - 根据槽位 2D 坐标嵌入和相邻已就位邻居的特征计算与打乱碎片的相似度匹配得分（Logits）。
   - 选块填入网格，调用 `LocalFusion` 执行局部十字注意力交互。
4. **双层注意力**：
   - `LocalFusion`：3x3 十字掩码卷积 + 局部十字 Transformer 块。
   - `GlobalFusion`：整网格 2D 可学习位置编码 + 2 层全局 Transformer 聚合。
5. **彻底移除边缘卷积**：
   - 无 6 通道边带，无法向梯度计算，无边缘卷积网络，极度轻量快速。
6. **纯粹的位置预测分类损失**：
   - 槽位多分类交叉熵损失。
   - 监控 Patch 级还原命中率（Patch-level Accuracy）和整图完美复原率（Image-level Perfect Reconstruction Rate）。
   - 训练每轮输出训练集与验证集的损失与命中率。

---

## 快速使用说明

### 1. 运行单元测试
```bash
cd /home/cjc/桌面/myidea
PYTHONPATH=. pytest v1/tests/test_v1.py -v
```

### 2. 启动训练

训练 $3\times3$ 网格（默认）：
```bash
cd /home/cjc/桌面/myidea
PYTHONPATH=. python -m v1.train \
  --grid_size 3 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 50 \
  --save_dir ./checkpoints
```

训练 $5\times5$ 网格：
```bash
PYTHONPATH=. python -m v1.train \
  --grid_size 5 \
  --batch_size 64 \
  --lr 3e-4 \
  --epochs 50 \
  --save_dir ./checkpoints
```

### 3. 评估模型
```bash
PYTHONPATH=. python -m v1.evaluate \
  --checkpoint ./checkpoints/grid3/best_checkpoint.pth \
  --batch_size 64
```

### 4. 拼图还原效果可视化
```bash
PYTHONPATH=. python -m v1.visualize \
  --checkpoint ./checkpoints/grid3/best_checkpoint.pth \
  --output_path ./visualizations/grid3_result.png \
  --num_samples 5
```
生成的对比图包含：**原始原图 vs 打乱后碎片堆 vs 模型还原拼图结果**。
