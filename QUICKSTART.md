# Quick Start Guide

## 快速开始指南

这个项目实现了 English → Chinese 机器翻译，包含 LSTM、GRU 和 Transformer 三个模型。

## 安装依赖

```bash
cd /home/grj/hw/CS5489/MT
pip install -r requirements.txt
```

## 运行完整流程

### 方法 1: 使用自动化脚本（推荐）

```bash
python run_all.py
```

这个脚本会自动执行：
1. 下载数据
2. 数据预处理
3. 数据探索
4. 训练所有模型
5. 评估模型
6. 生成可视化

### 方法 2: 逐步执行

#### Step 1: 下载数据
```bash
python data/download_data.py
python data/prepare_iwslt17.py
```

#### Step 2: 数据探索
```bash
python src/data_exploration.py
```

这会生成：
- `figures/length_distribution.png` - 句子长度分布
- `figures/word_frequency.png` - 词频统计
- `figures/alignment_examples.txt` - 对齐示例

#### Step 3: 训练模型

训练 LSTM:
```bash
python scripts/train.py --model lstm --config config.yaml
```

训练 GRU:
```bash
python scripts/train.py --model gru --config config.yaml
```

训练 Transformer:
```bash
python scripts/train.py --model transformer --config config.yaml
```

#### Step 4: 评估模型

```bash
# 评估 LSTM
python scripts/evaluate.py --model lstm --checkpoint checkpoints/lstm/lstm_best.pt --config config.yaml

# 评估 GRU
python scripts/evaluate.py --model gru --checkpoint checkpoints/gru/gru_best.pt --config config.yaml

# 评估 Transformer
python scripts/evaluate.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --config config.yaml
```

#### Step 5: 可视化

```bash
# 生成 t-SNE 可视化
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task tsne

# 生成聚类可视化
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task cluster

# 生成训练曲线
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task curves

# 生成所有可视化
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task all
```

## 输出文件说明

### 训练输出
- `checkpoints/{model}/{model}_best.pt` - 最佳模型检查点
- `checkpoints/{model}/{model}_latest.pt` - 最新检查点
- `checkpoints/{model}/history.json` - 训练历史（损失曲线数据）
- `checkpoints/{model}/training_log.txt` - 详细训练日志

### 评估输出
- `results/{model}_bleu.json` - BLEU 分数
- `results/{model}_examples.txt` - 翻译示例

### 可视化输出
- `figures/{model}_tsne.png` - t-SNE 可视化
- `figures/{model}_clusters.png` - KMeans 聚类
- `figures/{model}_training_curves.png` - 训练曲线

## 配置说明

主要配置在 `config.yaml` 中：

- **数据设置**: 最大长度、最小长度
- **预处理**: 选择 tokenization 方式（`word` 或 `bpe`）、词汇表大小
  - `word`: 词级别分词，适合快速实验
  - `bpe`: 子词分词，更好地处理OOV问题（推荐）
- **模型设置**: 隐藏层大小、层数、dropout 等
- **训练设置**: batch size、学习率、epochs、label smoothing 等
- **评估设置**: beam size、最大长度、是否使用beam search等
- **交叉验证**: CV折数、每个fold的训练epoch数

## 常见问题

### Q: 训练很慢怎么办？
A: 
- 减小 `batch_size`（但可能影响效果）
- 减小 `max_length`
- 使用 GPU（如果有的话会自动使用）

### Q: 内存不足？
A:
- 减小 `batch_size`
- 减小 `vocab_size`
- 减小模型大小（hidden_dim, num_layers）

### Q: 如何修改超参数？
A: 编辑 `config.yaml` 文件，然后重新训练。或者使用 `train_with_cv.py` 进行交叉验证自动选择最佳超参数。

### Q: 如何使用交叉验证？
A: 
```bash
python scripts/train_with_cv.py --model lstm --config config.yaml --n_folds 3
```
这会在训练集上进行3折交叉验证，自动选择最佳超参数。

### Q: 如何切换tokenization方式？
A: 在 `config.yaml` 中修改 `preprocessing.tokenization` 为 `"word"` 或 `"bpe"`，然后重新训练。

### Q: 如何添加新的模型？
A: 
1. 在 `src/models/` 中创建新模型文件
2. 在 `scripts/train.py` 的 `build_model` 函数中添加模型构建逻辑
3. 在 `scripts/evaluate.py` 的 `load_model` 函数中添加模型加载逻辑

## 报告撰写

完成实验后，编辑 `report/report_template.md`，填入你的实验结果：

1. 数据探索结果（从 `figures/` 获取图表）
2. 模型性能（BLEU 分数）
3. 翻译示例（从 `results/` 获取）
4. 可视化结果（从 `figures/` 获取）

## 项目结构

```
MT/
├── data/              # 数据下载和预处理
├── src/               # 源代码
│   ├── models/        # 模型定义
│   ├── preprocessing.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── visualization.py
├── scripts/           # 训练和评估脚本
├── checkpoints/       # 模型检查点（训练后生成）
├── results/           # 评估结果（评估后生成）
├── figures/           # 可视化图表（生成后）
└── report/            # 报告模板
```

## 下一步

1. 运行完整流程：`python run_all.py`
2. 查看结果和图表
3. 填写报告模板
4. 提交作业！

祝你好运！🎉

