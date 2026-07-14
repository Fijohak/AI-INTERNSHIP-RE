# AI 科研实践线上实习 — 8 周任务清单

> 来源：飞书文档《AI 科研实践线上实习任务书 v2.2》
> 仓库命名：`AI-Internship-你的姓名`

---

## 基础准备

- [ ] 建立 GitHub / Gitee 个人项目仓库（已有：`kaggle_practice`）
- [ ] 建立 Kaggle 账号并完善个人主页
- [ ] 按周建立项目目录结构
- [ ] 每周按要求提交代码、图表、实验结果和报告
- [ ] 最终形成一篇完整技术文章 + 答辩 PPT

---

## 每周目录统一模板

```
WeekXX_ProjectName/
├── README.md
├── notebooks/
├── src/
├── data_source.md
├── results/
│   ├── figures/
│   ├── metrics.csv
│   └── screenshots/
├── error_analysis.md
├── report.md
└── run_log.txt
```

---

## 第 1 周：AI 认知基础 — Python / Iris 分类 / 线性回归 🟢

**课程学习：**
- [ ] 吴恩达 AI for Everyone（认知 AI 概念）
- [ ] 吴恩达 AI Prompting for Everyone

**Iris 分类项目：**
- [ ] 加载 Iris 数据集（sklearn 内置）
- [ ] 数据探索 & 可视化（散点图、箱线图）
- [ ] 实现 KNN / Logistic Regression / SVM 分类
- [ ] 模型对比：准确率、混淆矩阵、分类报告
- [ ] 撰写 Iris 项目 report.md

**线性回归项目：**
- [ ] 选择一个回归数据集（California Housing / Diabetes）
- [ ] 数据预处理 & 特征工程
- [ ] 实现 Linear Regression（sklearn / 手动）
- [ ] 评估：R²、MSE、残差分析
- [ ] 撰写线性回归 report.md

**本周提交：**
- [ ] GitHub/Kaggle 初始化截图
- [ ] Iris 分类代码 + 图表 + 报告
- [ ] 线性回归代码 + 图表 + 报告

---

## 第 2 周：传统机器学习实战专题 🟡

**必做项目（至少 2 个）：**

| 项目 | 数据集 | 算法 |
|------|--------|------|
| 信用卡违约预测 | [UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) | LR, DT, RF, XGBoost |
| 糖尿病预测 | [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) | 同上 |

**选做项目（至少完成 2 个）：**
- [ ] 学生成绩预测 — [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance)
- [ ] 用户行为分析 — [Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- [ ] 异常行为识别 — Isolation Forest / One-Class SVM
- [ ] KMeans / PCA / 层次聚类（选一个聚类项目）

**本周提交：**
- [ ] ≥ 4 个 ML 项目的完整代码
- [ ] 多模型对比表格 & 图表
- [ ] feature importance / SHAP 分析
- [ ] 每个项目的 report.md

---

## 第 3 周：Kaggle 表格竞赛专题 🟠

**课程学习：**
- [ ] 吴恩达 Generative AI for Everyone

**Kaggle 竞赛：**

| 竞赛 | 链接 | 算法 |
|------|------|------|
| Titanic 生存预测 | [Kaggle Titanic](https://www.kaggle.com/c/titanic) | RF, XGBoost, 集成 |
| House Prices 房价预测 | Kaggle House Prices | 回归/分类, RF, XGBoost |

- [x] 完成 Titanic 特征工程、模型训练、Kaggle 提交
- [x] 完成 House Prices 特征工程、模型训练、Kaggle 提交
- [ ] 学习 Kaggle Notebook 的最佳实践
- [ ] 对比不同模型的 Kaggle Score

**本周提交：**
- [ ] 2 个 Kaggle 项目完整代码 + Notebook
- [ ] Kaggle 提交截图（含排名/分数）
- [ ] 多模型对比 & 特征重要性图
- [ ] 每个项目的 report.md

---

## 第 4 周：论文复现① — LeNet-5 + MNIST 🔵 **← 当前**

> 当前项目：`Week4-Lenet5-MINIST/` ✅ 已完成

- [x] 创建项目骨架
- [x] 论文阅读笔记（LeCun 1998）
- [x] 模型定义 `src/model.py`
- [x] 数据加载 `src/dataset.py`
- [x] 训练脚本 `src/train.py`
- [x] 评估脚本 `src/evaluate.py`
- [x] 可视化 `src/visualize.py`
- [x] 实验对比（AvgPool vs MaxPool、SGD vs Adam、Tanh vs ReLU、Seed 控制）
- [x] 补齐 `data_source.md`
- [x] 整理 `error_analysis.md`（汇总所有实验的错误分析）
- [x] 撰写 `report.md`（最终实验报告，综合所有实验发现）
- [x] 整理 `run_log.txt`

**本周提交：**
- [ ] PyTorch 完整可运行代码
- [ ] 训练曲线 + 准确率 ≥ 98% 截图
- [ ] 论文复现报告（与本周期刊/会议报告格式一致）
- [ ] 完整实验矩阵（参数对比、消融实验）

---

## 第 5 周：ResNet 论文复现 + Transformer 启动 🟣

**ResNet 复现：**
- [ ] 阅读 ResNet 论文（He et al., Deep Residual Learning）
- [ ] 在 CIFAR-10 上复现 ResNet（至少 ResNet-18/34）
- [ ] 训练曲线、准确率、复现报告
- [ ] 消融实验：有无残差连接的对比

**Transformer 初读：**
- [ ] 阅读《Attention Is All You Need》（已有 PDF）
- [ ] 整理 Transformer 结构笔记（Self-Attention、Multi-Head、Position Encoding）
- [ ] 初读笔记 & 疑问记录

**本周提交：**
- [ ] ResNet 代码 + 训练曲线 + 复现报告
- [ ] 消融实验（残差连接对比）
- [ ] Transformer 初步笔记

---

## 第 6 周：Dog Breed 图像识别 + 人脸识别进阶 🔴

**Dog Breed 迁移学习：**
> 现有目录：`dog_breed_identify/`

- [ ] 数据集准备（Stanford Dogs / Kaggle Dog Breed）
- [ ] 使用预训练模型（ResNet / EfficientNet）做迁移学习
- [ ] Top-1 / Top-5 准确率评估
- [ ] Kaggle 提交（如有对应竞赛）

**人脸识别进阶（选做）：**
- [ ] 选择一个人脸识别任务（Face Verification / 表情识别）
- [ ] 实现 or 调用现有模型
- [ ] 评估 & 报告

**本周提交：**
- [ ] Dog Breed 代码 + 训练曲线 + Kaggle 提交截图
- [ ] Top-1 / Top-5 准确率
- [ ] 人脸识别选做项目（如完成）

---

## 第 7 周：Transformer 精读 + 本地大模型部署 + RAG 🟤

**Transformer 精读：**
- [ ] 逐段精读 Transformer 论文，撰写详细笔记
- [ ] 手动实现 Self-Attention / Multi-Head Attention（NumPy/PyTorch）
- [ ] 理解 Q、K、V 的物理含义

**本地大模型部署：**
- [ ] 使用 Ollama / llama.cpp 部署一个本地 LLM（如 Qwen2.5）
- [ ] 测试基本推理能力
- [ ] 截图 & 笔记

**RAG 论文问答项目：**
- [ ] 搭建 RAG 系统（LangChain / LlamaIndex + 向量数据库）
- [ ] 上传一篇论文 PDF，实现问答
- [ ] 评估检索准确率和回答质量

**本周提交：**
- [ ] Transformer 精读笔记 + 手动实现代码
- [ ] 本地模型部署截图 + 推理测试
- [ ] RAG Demo 截图 + 问答效果评估

---

## 第 8 周：最终整合、技术文章与答辩 ⚫

**技术文章：**
- [ ] 按照最终文章结构撰写
- [ ] 整合 1-7 周的核心成果
- [ ] 图文并茂（图表、代码片段、实验结果）
- [ ] 结构：摘要 → 引言 → 方法 → 实验 → 分析 → 结论

**答辩 PPT：**
- [ ] 项目全景概览（1 页）
- [ ] 每周核心成果展示（7-8 页）
- [ ] 关键技术亮点 & 创新点
- [ ] 遇到的困难 & 解决方案
- [ ] 个人收获 & 未来方向

**仓库整理：**
- [ ] 整理总 README.md（链接各周项目）
- [ ] 统一代码风格 & 注释
- [ ] 检查所有链接是否有效
- [ ] 完善总实验汇总表

**本周提交：**
- [ ] 完整技术文章（Word/PDF + Markdown）
- [ ] 答辩 PPT（含图表和可视化）
- [ ] 完整 GitHub 仓库（所有周的代码 + 报告）
- [ ] 总实验表（汇总 8 周的所有模型和结果）

---

## 结业标准

| 等级 | 要求 |
|------|------|
| **合格** | 每周按时提交，代码可运行，报告基本完整 |
| **良好** | 所有必做项目完成 + 至少 2 个选做项目，实验设计合理 |
| **优秀** | 全部项目完成 + 实验对比深入 + 技术文章质量高 + 答辩表现好 |

---

## 每周验收 checklist

每周提交时必须包含：
- [ ] `README.md` — 本周项目说明
- [ ] `src/` or `notebooks/` — 可运行代码
- [ ] `results/figures/` — 关键图表
- [ ] `results/metrics.csv` — 模型指标汇总
- [ ] `error_analysis.md` — 错误分析
- [ ] `report.md` — 实验报告
- [ ] `run_log.txt` — 运行日志

---

## 学习资源

| 资源 | 用途 | 状态 |
|------|------|------|
| 吴恩达 AI for Everyone | 第 1 周 | 📋 |
| 吴恩达 Generative AI for Everyone | 第 3 周 | 📋 |
| 吴恩达 AI Prompting for Everyone | 第 1 周 | 📋 |
| Andrej Karpathy 大模型入门 | 第 7 周 | 📋 |
| LeNet-5 论文 (LeCun 1998) | 第 4 周 | ✅ 已读 |
| ResNet 论文 (He et al.) | 第 5 周 | 📋 |
| Attention Is All You Need | 第 5/7 周 | 📋 |
| NLP-LLM-Chapter-2-3-Transformer | 第 5/7 周 | 📋 |

---

> 最后更新：2026-07-15
> 当前进度：Week 4 已完成，下一步 Week 5（ResNet 复现 + Transformer）
