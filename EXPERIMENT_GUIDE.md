# 完整实验指南 - 按照 Rubric 满分路线

本指南按照 Rubric 要求，提供完整的实验执行步骤。

## 📋 实验流程概览

### 阶段 1: 数据探索和预处理

```bash
# 1. 下载/处理数据
python data/download_data.py

# 2. 数据探索
python src/data_exploration.py
```

### 阶段 2: Cross-Validation 超参调优（在 train 集上）

```bash
# 对 LSTM + BPE 做 3-fold CV 选超参
python scripts/train_with_cv.py --model lstm --config config.yaml --n_folds 3
```

这会：
- 在 train 集上做 3-fold CV
- 测试多个超参数组合（lr, hidden_size, dropout）
- 选择最佳超参数
- 保存 CV 结果到 `checkpoints/cv_results/`

### 阶段 3: 主实验（使用最佳超参）

#### 3.1 训练不同模型（都用 BPE）

```bash
# LSTM + BPE
python scripts/train.py --model lstm --config config.yaml

# GRU + BPE  
python scripts/train.py --model gru --config config.yaml

# Transformer + BPE
python scripts/train.py --model transformer --config config.yaml
```

#### 3.2 对比不同特征（Word vs BPE）

```bash
# 修改 config.yaml: preprocessing.tokenization = "word"
python scripts/train.py --model lstm --config config.yaml

# 修改 config.yaml: preprocessing.tokenization = "bpe"  
python scripts/train.py --model lstm --config config.yaml
```

### 阶段 4: Extra 实验

#### 4.1 Beam Search vs Greedy

```bash
# Greedy (默认)
python scripts/evaluate.py --model lstm --checkpoint checkpoints/lstm/lstm_best.pt

# Beam Search (修改 config.yaml: evaluation.use_beam_search = true)
python scripts/evaluate.py --model lstm --checkpoint checkpoints/lstm/lstm_best.pt
```

#### 4.2 Label Smoothing

```bash
# 修改 config.yaml: training.label_smoothing = 0.1
python scripts/train.py --model lstm --config config.yaml
```

#### 4.3 可视化（DimRed/Clustering）

```bash
# t-SNE 可视化
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task tsne

# KMeans 聚类
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task cluster

# 训练曲线
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task curves
```

### 阶段 5: 结果汇总

```bash
# 生成所有结果的汇总
python scripts/summarize_results.py

# 这会生成：
# - results/results_table.csv
# - results/summary_report.md
# - figures/model_comparison.png
```

## 📊 Rubric 对照检查清单

### ✅ Methods (15分)
- [x] LSTM Seq2Seq + Attention
- [x] GRU Seq2Seq + Attention  
- [x] Transformer Encoder-Decoder

### ✅ Experiment Setup (10分)
- [x] 3-fold CV 在 train 集上调超参
- [x] 使用独立 valid 集做 early stopping
- [x] 只在 test 集上评估一次
- [x] 报告 CV 的 mean ± std

### ✅ Features (10分)
- [x] Word-level tokenization
- [x] BPE subword tokenization

### ✅ DimRed/Clustering (5分)
- [x] t-SNE 可视化 encoder embeddings
- [x] KMeans 聚类分析
- [x] 训练曲线可视化

### ✅ Extra Data (5分)
- [x] 句子长度分布
- [x] 词频统计
- [x] 训练过程可视化

### ✅ Extra Method (10分)
- [x] Beam Search 解码
- [x] Label Smoothing

### ✅ Extra Features (5分)
- [x] BPE 作为新特征表示

### ✅ Extra Justification (5分)
- [ ] 在报告中写清楚设计理由

## 🔧 配置文件说明

### config.yaml 关键设置

```yaml
# 特征选择
preprocessing:
  tokenization: "bpe"  # "word" 或 "bpe"
  vocab_size: 8000

# 训练设置
training:
  label_smoothing: 0.0  # 设为 0.1 启用 label smoothing

# 评估设置
evaluation:
  use_beam_search: false  # 设为 true 使用 beam search
  beam_size: 5

# Cross-validation
cross_validation:
  n_folds: 3
  cv_epochs: 5  # CV 时每个 fold 训练几个 epoch
```

## 📝 报告撰写要点

### 1. Methods 部分
- 描述三个模型的架构
- 说明 attention 机制
- 对比 RNN vs Transformer

### 2. Experiment Setup 部分
- **必须写清楚**：
  - "We use 3-fold cross-validation on the training set to tune hyperparameters"
  - "We report mean ± std validation loss across folds"
  - "After CV, we train on full training set and evaluate once on test set"

### 3. Features 部分
- 对比 word-level vs BPE
- 说明 BPE 的优势（OOV 处理）

### 4. Results 部分
- 主表：不同模型 + 不同特征的 BLEU/loss
- CV 结果表：超参组合的 mean ± std
- 训练曲线图
- t-SNE 可视化图

### 5. Extra Experiments 部分
- Beam Search vs Greedy 的 BLEU 对比
- Label Smoothing 的影响
- DimRed/Clustering 的发现

### 6. Discussion 部分
- 错误分析（bad cases）
- 模型优缺点对比
- 训练速度 vs 性能权衡

## 🚀 快速开始

### 完整实验流程（推荐顺序）

```bash
# 1. 数据准备
python data/download_data.py
python src/data_exploration.py

# 2. CV 调超参（选一个代表性模型，如 LSTM+BPE）
python scripts/train_with_cv.py --model lstm --n_folds 3

# 3. 主实验（用最佳超参）
python scripts/train.py --model lstm
python scripts/train.py --model gru
python scripts/train.py --model transformer

# 4. 特征对比（Word vs BPE）
# 修改 config.yaml: tokenization = "word"
python scripts/train.py --model lstm

# 5. Extra 实验
# Beam Search
python scripts/evaluate.py --model lstm --checkpoint checkpoints/lstm/lstm_best.pt
# (修改 config.yaml: use_beam_search = true 再跑一次)

# Label Smoothing
# (修改 config.yaml: label_smoothing = 0.1)
python scripts/train.py --model lstm

# 6. 可视化
python scripts/visualize.py --model transformer --checkpoint checkpoints/transformer/transformer_best.pt --task all

# 7. 结果汇总
python scripts/summarize_results.py
```

## 📈 预期结果

完成所有实验后，你应该有：

1. **训练历史**：`checkpoints/{model}/history.json`
2. **CV 结果**：`checkpoints/cv_results/cv_summary.json`
3. **评估结果**：`results/{model}_bleu.json`
4. **可视化**：`figures/{model}_*.png`
5. **汇总报告**：`results/summary_report.md`

## 💡 提示

- CV 阶段可以训练少一些 epoch（如 5 个）以节省时间
- 主实验用完整 epoch 数（如 20 个）
- 保存所有实验结果，方便写报告时引用
- 多做几个翻译例子展示，包括好案例和坏案例

祝实验顺利！🎉

