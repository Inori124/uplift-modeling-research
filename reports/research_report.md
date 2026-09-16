# 从转化预测到增量响应排序：Criteo 随机实验上的 Uplift Modeling 研究

**版本：** v2.1
**数据：** Criteo Uplift Modeling 公共随机实验数据
**研究类型：** 可复现的离线方法比较与稳健性分析
**状态：** 论文初稿；结果仅用于研究，不代表线上投放收益

## 摘要

互联网增长策略通常需要决定“触达哪些用户”，而不是单纯预测“哪些用户会转化”。高转化概率用户可能在没有广告、Push 或优惠券时也会完成目标行为；对这类用户进行触达会消耗预算，却未必产生增量。Uplift modeling 将目标从条件转化概率扩展为个体处理效应，即给定处理前特征时，接受与不接受处理的潜在结果平均差异。

本文使用 Criteo 随机 treatment/control 实验数据，比较 T-learner、S-learner 与 X-learner 在增量响应排序任务上的表现。研究只使用 12 个处理前匿名特征，避免将处理后变量引入模型。主实验固定开发集划分，统一使用未加权 HistGradientBoosting 结果模型，X-learner 使用 3-fold training-only cross-fitting；使用 IPW AUUC、Qini area 和 Top-K policy gain 评估。主实验重复 5 个模型随机种子，并在冻结配置后使用与开发集按行号不重叠的 300,000 行 holdout 做一次性评估。

在最终 holdout 上，S-learner、X-learner 和 T-learner 的 Qini area 分别为 `0.000419`、`0.000375` 和 `0.000285`；随机排序参考为 `0.000031`。在当前数据、特征和模型配置下，S-learner 表现最好，X-learner 次之。该结论只适用于公开实验数据上的离线排序，不可直接解释为线上 ROI、收入提升或投放策略上线效果。

## 1. 引言

### 1.1 业务动机

假设平台可以向用户展示广告或发送 Push。传统转化模型会估计：

```text
P(Y=1 | X)
```

其中 `X` 是用户特征，`Y` 是转化结果。这个目标适合预测需求，却不能区分以下四类用户：

- **Sure things：** 不触达也会转化；
- **Persuadables：** 触达后才更可能转化；
- **Sleepers：** 触达几乎没有影响；
- **Lost causes：** 无论是否触达都不会转化。

如果按照转化概率排序，平台可能把预算优先分配给第一类用户。增长策略真正希望识别的是第二类用户，即具有正向增量响应的人群。因此本文关注：

> 在随机实验数据中，模型能否把更可能因触达而增加转化的用户排在前面？

### 1.2 研究问题

本文回答四个问题：

1. T-learner、S-learner 和 X-learner 能否在稀疏转化数据上产生高于随机排序的增量响应排序？
2. 统一结果模型和未加权概率估计后，三种 learner 的排序效果如何？
3. X-learner 的 cross-fitting 和处理组构成变化会如何影响结果？
4. 开发集上的发现能否在未参与模型选择的 holdout 上保持方向一致？

## 2. 数据

### 2.1 数据来源与字段

数据来自 Criteo Uplift Modeling 公共随机实验。完整文件包含 **13,979,592 行**，每行包括：

| 字段 | 含义 | 是否进入模型 |
| --- | --- | --- |
| `f0`–`f11` | 匿名处理前用户特征 | 是 |
| `treatment` | 是否被分配到处理组 | 处理变量 |
| `conversion` | 是否发生转化 | 结果变量 |
| `visit` | 访问相关结果字段 | 否 |
| `exposure` | 处理/曝光相关字段 | 否 |

`visit` 和 `exposure` 不进入模型输入，因为它们可能包含处理之后的信息。完整 CSV 约 3.25 GB，不上传 GitHub；仓库只保存代码、汇总结果、图表和说明。

### 2.2 全量数据的描述性统计

全量 treatment 比例约为 `0.8500`，conversion 比例约为 `0.002917`。分组描述性统计如下：

| 分组 | 样本数 | 转化率 | 访问率 | 曝光率 |
| --- | ---: | ---: | ---: | ---: |
| Control | 2,096,937 | 0.1938% | 3.8201% | 0% |
| Treatment | 11,882,655 | 0.3089% | 4.8543% | 3.6037% |

处理组与对照组的未调整转化率差为约 `0.1152` 个百分点。该差异描述总体平均处理效果，不能直接说明哪些个体具有正向 uplift，也不能替代模型排序评估。

![全量数据中的处理分配与转化率](figures/paper_data_overview.png)

**图 1.** Treatment/control 样本规模和 conversion rate。左图使用行数，右图使用比例；两个面板采用不同纵轴尺度。

## 3. 方法

### 3.1 条件平均处理效应（CATE）

令 `X` 表示处理前特征，`T∈{0,1}` 表示处理分组，`Y∈{0,1}` 表示转化结果。目标是估计条件平均处理效应（CATE）：

```text
τ(x) = E[Y(1) - Y(0) | X=x]
     = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)
```

在随机化、正值处理概率、稳定处理版本（SUTVA）以及处理前特征不受 treatment 影响等假设下，treatment/control 的组间差异具有因果解释基础。单个用户的两个潜在结果不能同时观测，因此本文估计的是条件平均处理效应及其排序，而不是观测每个用户的真实个体效应。

### 3.2 T-learner

T-learner 分别训练两个结果模型：

```text
μ1(x) = P(Y=1 | X=x, T=1)
μ0(x) = P(Y=1 | X=x, T=0)
τ̂(x) = μ̂1(x) - μ̂0(x)
```

本文使用两个未加权 `HistGradientBoostingClassifier`。

### 3.3 S-learner

S-learner 使用一个统一结果模型，把 treatment 作为输入特征：

```text
μ(x, t) = P(Y=1 | X=x, T=t)
τ̂(x) = μ̂(x, 1) - μ̂(x, 0)
```

该方法参数共享程度更高，在 treatment/control 样本比例不平衡时可能具有更低的估计方差，但也可能将 treatment 与特征交互压缩进同一个模型。

### 3.4 X-learner

X-learner 先拟合结果模型，再用对侧结果构造伪处理效应：

```text
d1 = Y - μ̂0(X),  treatment group
 d0 = μ̂1(X) - Y, control group
```

随后分别拟合 `τ1(X)` 与 `τ0(X)`，并以训练样本 treatment 比例 `p_train` 组合：

```text
τ̂(x) = (1-p_train)τ̂1(x) + p_trainτ̂0(x)
```

本文用 3-fold training-only cross-fitting 生成 `μ̂0` 和 `μ̂1` 的 out-of-fold 预测，减少结果模型过拟合对伪处理效应的影响。cross-fitting 不是对所有因果识别问题的修复；随机实验设计、特征时间边界和评估集隔离仍然是必要条件。

## 4. 实验设计

### 4.1 开发集与验证集

从全量数据中按固定 `seed=2027` 均匀抽取 1,000,000 行开发集，再按固定 `split_seed=2027` 分为：

- **Train pool：** 750,000 行；
- **Development validation：** 250,000 行。

模型使用 2027–2031 五个随机种子。所有模型使用同一份验证集，模型之间不共享预测结果，但共享切分和评估口径。

### 4.2 固定训练量的构成实验

为了区分“训练集总量变化”和“处理组构成变化”，另固定训练样本数为 120,000，只改变 treated fraction：`0.85`、`0.50`、`0.33`、`0.20`。每个设定运行 5 个 seed，抽样只依赖 treatment，不使用 conversion 或模型结果进行筛选。

### 4.3 最终 holdout

在模型配置和研究问题冻结后，从原始全量数据中额外抽取 300,000 行，与开发集按原始行号不重叠。完整开发集训练固定 v2.1 配置，holdout 只评估一次，不参与模型选择。

## 5. 评估指标

### 5.1 IPW 增量贡献

验证集的 treatment 分配概率使用 `p=mean(T)` 作为 plug-in。每个样本的 IPW 增量贡献为：

```text
z_i = T_i Y_i / p - (1-T_i)Y_i/(1-p)
```

按预测 uplift 降序排列后，前 `q` 比例人群的累计 gain 定义为：

```text
G(q) = (1/N) Σ_{i in top-q} z_i
```

`G(q)` 的分母是整个评估集 `N`，所以它是每个总体用户贡献的策略价值。若要表示被选中人群内部平均增量，需要计算 `G(q)/q`。

### 5.2 AUUC 与 Qini area

```text
AUUC = ∫ G(q) dq
Qini(q) = G(q) - qG(1)
Qini area = ∫ Qini(q) dq
```

Qini area 扣除了随机选择在同一预算比例下的线性基线。理想情况下，Qini 曲线在前段显著高于 0，表示模型能把增量人群排在前面。

### 5.3 Bootstrap 区间

使用按 treatment 分层的 paired percentile bootstrap。每次重抽样在相同的验证集行上对所有模型使用同一组重复计数，因此模型差异是配对的。该区间：

- 固定已训练模型的预测分数；
- 不重新训练模型；
- 不包含模型选择和超参数搜索不确定性；
- 不替代最终 holdout 泛化评估。

## 6. 开发集结果

### 6.1 Qini 曲线

![开发集 Qini 曲线](figures/paper_qini_validation.png)

**图 2.** Seed 2027 开发验证集上的 Qini 曲线。横轴是触达用户比例，纵轴是相对总体用户归一化的累计增量。T/S/X 曲线在前段高于随机基线；曲线面积用于汇总不同预算比例下的排序表现。

5 个 seed 的主结果如下：

| 方法 | Qini area 均值 ± 标准差 | Top20 policy gain 均值 ± 标准差 |
| --- | ---: | ---: |
| Random draw | 0.000005 ± 0.000047 | 0.000274 ± 0.000115 |
| T-learner HGB | 0.000412 ± 0.000037 | 0.001080 ± 0.000048 |
| S-learner HGB | **0.000518 ± 0.000087** | **0.001205 ± 0.000118** |
| X-learner HGB + cross-fitting | 0.000427 ± 0.000070 | 0.001060 ± 0.000066 |

在统一未加权 HGB 设定下，S-learner 的平均 Qini area 和 Top20 policy gain 最高。X-learner 的平均 Qini area 为正，但低于 S-learner；这与早期类别加权、不同基础学习器组合得到的“X-learner 最优”现象不同，说明模型结构、基础学习器和概率校准需要分开讨论。

### 6.2 处理组构成稳健性

![处理组构成稳健性](figures/paper_treatment_fraction.png)

**图 3.** 固定训练量 120,000 时，X-learner Qini area 随 treated fraction 的变化。点为 5 个 seed 均值，误差条为 seed 间标准差。

| Treated fraction | X-learner Qini area 均值 | 标准差 |
| ---: | ---: | ---: |
| 0.85 | 0.000259 | 0.000198 |
| 0.50 | 0.000342 | 0.000045 |
| 0.33 | 0.000312 | 0.000068 |
| 0.20 | 0.000159 | 0.000111 |

该结果没有呈现简单的单调趋势。`0.20` 条件下均值较低，但 `0.50` 和 `0.33` 高于 `0.85`；因此不能将原始 treatment 比例或某一比例宣称为普遍最优。实验只描述当前固定训练量、模型配置和数据抽样下的离线稳健性。

## 7. 最终 holdout 结果

### 7.1 点估计与条件区间

![最终 holdout Qini area](figures/paper_holdout_qini_ci.png)

**图 4.** 冻结模型在最终 holdout 上的 Qini area。点为点估计，误差条为固定模型分数条件下的 95% paired bootstrap percentile CI。

| 方法 | Qini area | Top20 policy gain | Qini 95% 条件 bootstrap CI |
| --- | ---: | ---: | ---: |
| Random draw | 0.000031 | 0.000221 | [-0.000111, 0.000153] |
| T-learner HGB | 0.000285 | 0.000732 | [0.000067, 0.000501] |
| S-learner HGB | **0.000419** | **0.000887** | [0.000199, 0.000627] |
| X-learner HGB + cross-fitting | 0.000375 | 0.000838 | [0.000146, 0.000560] |

Holdout 上 S-learner 的点估计最高，X-learner 次之，T-learner 再次。三种模型的点估计均高于随机参考，且各自相对理论随机期望的条件区间位于 0 以上；但模型间的 paired difference 区间跨越 0（例如 S−X 的 Qini difference CI 为约 `[-0.000200, 0.000242]`），因此不能宣称模型之间存在确定性优劣。这里的“高于随机”是当前 holdout 和当前冻结分数下的统计描述，不等于线上策略显著性或收入提升。

![最终 holdout Qini 曲线](figures/paper_qini_holdout.png)

**图 5.** 最终 holdout 的 Qini 曲线。相同评估集上的曲线可以观察不同预算比例下的累计增量差异；S-learner 在多数前段比例保持较高，X-learner 在部分低预算区间接近或超过 S-learner。

### 7.2 结果解读

开发集和 holdout 的模型排序方向一致：S-learner 位于第一，X-learner 位于第二。该一致性提高了当前实验结论的可信度，但仍有三个边界：

1. holdout 与开发集来自同一公开数据分布，不是跨时间或跨业务线外部验证；
2. holdout 只验证了冻结的一组超参数，没有覆盖模型选择本身的不确定性；
3. Criteo 的匿名特征和实验处理不等价于具体公司的 Push、广告或优惠券业务。

## 8. 类别加权敏感性

早期实验同时比较了 `class_weight=None` 和 `class_weight='balanced'`。由于 conversion 极为稀疏，`balanced` 会改变训练中类别先验，使 `predict_proba` 更适合分类边界而不一定保持真实概率校准。因而：

- 未加权概率模型作为 v2 主结果；
- 类别加权只作为敏感性分析；
- 不能把类别加权版本与未加权版本之间的差异直接解释成 learner 结构差异。

相关图表见 [`class_weight_sensitivity.png`](figures/class_weight_sensitivity.png)。

## 9. 局限性

### 9.1 处理概率

当前评估使用验证集 treatment 频率作为 plug-in 概率。正式复现时应核对数据发布方对随机分配概率的定义，并在可用时使用设计概率。X-learner 的效应组合使用训练样本 treatment 频率，它描述的是当前训练设计，不必然等于目标人群的部署 propensity。

### 9.2 匿名特征

`f0`–`f11` 没有业务语义，无法判断某个模型是否利用了活跃度、渠道或用户价值等可解释属性。本文只讨论排序效果，不声称发现了可直接执行的人群规则。

### 9.3 稀疏转化

conversion 率约为 0.29%，对照组转化事件更少；seed2027 validation 中 control 仅有 68 次 conversion（37,400 行），treatment 有 671 次（212,600 行）。Qini、AUUC 和 Top-K policy gain 具有明显抽样波动；5 个 seed 的标准差和 bootstrap 区间应与点估计一起报告。

### 9.4 离线到线上

没有触达成本、收益单价和线上随机实验时，policy gain 只能解释为离线增量转化估计。不能将其改写成 ROI、收入提升、节省预算或上线收益。

## 10. 可复现性

仓库提供：

- `src/run_v2_experiment.py`：单次 v2 实验；
- `src/run_study_v2.py`：主实验与固定训练量实验编排；
- `src/evaluation_v2.py`：tie-safe 指标和 paired bootstrap；
- `src/run_final_holdout.py`：冻结配置的一次性 holdout 评估；
- `src/plot_paper_figures.py`：本文图表生成；
- `tests/`：指标和实验流程测试；
- `results/`：汇总结果和最终 holdout 结果。

运行测试：

```bash
python3 -m unittest discover -s tests -v
```

运行图表：

```bash
MPLBACKEND=Agg python3 src/plot_paper_figures.py
```

重新运行完整开发实验：

```bash
PYTHONPATH=src python3 src/run_study_v2.py --suite all --workers 2 --bootstrap 200
```

## 11. 结论

在 Criteo 公开随机实验数据上，本文完成了从总体处理效果描述、uplift learner 建模、统一基础学习器比较、cross-fitting、固定训练量稳健性分析到最终 holdout 验证的实验闭环。冻结配置的 holdout 结果显示，S-learner 在当前离线 Qini area 和 Top20 policy gain 上最高，X-learner 次之，T-learner 再次，三者均高于随机排序参考。

该结论支持把项目写成“基于公开随机实验数据的用户增量响应建模研究”，但不支持声称线上 ROI 或商业收入提升。下一步若要继续投稿或用于求职，应核对 treatment assignment probability，补充不同样本规模和外部数据验证，并在报告中明确模型选择、holdout 与 bootstrap 的边界。
