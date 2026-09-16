# 用户增量响应建模研究报告（v2.1）

## 1. 研究问题

转化概率高的用户未必是触达后才转化。本文研究如何从随机 treatment/control 实验中估计用户的增量响应，并比较 T-learner、S-learner 和 X-learner 在稀疏转化与 treatment 不平衡条件下的离线排序效果。

## 2. 数据与变量

使用 Criteo uplift modeling 公共数据。全量数据有 13,979,592 行，12 个匿名处理前特征 `f0`–`f11`；`treatment` 比例约 0.85，`conversion` 比例约 0.0029。建模不使用 `visit`、`exposure` 或 `conversion` 以外的处理后信息。当前开发集为固定 seed=2027 的 1,000,000 行均匀抽样。

## 3. 实验设计

固定 `split_seed=2027`，75% 作为 train pool，25% 作为 validation。主实验使用模型 seed 2027–2031，所有方法使用相同的未加权 `HistGradientBoostingClassifier` 结果模型和 `HistGradientBoostingRegressor` 效应模型。X-learner 的 nuisance outcome 使用 3-fold training-only cross-fitting，以减少结果模型过拟合造成的伪效应偏差。评估集在模型和 seed 之间完全相同。

另设固定总训练量 120,000 的构成实验，treated fraction 为 0.85、0.50、0.33、0.20，每种 5 个 seed；抽样只依赖 treatment，不依赖结果变量。它检验的是训练构成和有限样本共同变化下的组合稳健性。

## 4. 评估指标

按预测 uplift 降序排列，使用验证集 `p=mean(treatment)` 作为 IPW plug-in 处理概率。对每名用户计算：

```text
z_i = T_i Y_i / p - (1-T_i)Y_i/(1-p)
```

累积 gain 按整个验证集人数归一化。报告 AUUC、Qini area、Top 10/20/30 policy gain；`topK_selected_ate` 是被选中人群内部的平均增量估计。报告的 paired bootstrap 是固定模型分数条件下的 95% 分位数区间，不包含重新训练或模型选择。

## 5. 结果

统一 HGB、未加权主实验 5 个 seed 的 Qini area：Random draw `0.000005±0.000047`，T-learner `0.000412±0.000037`，S-learner `0.000518±0.000087`，X-learner + cross-fitting `0.000427±0.000070`。Top20 policy gain 分别为 `0.000274`、`0.001080`、`0.001205`、`0.001060`。

在当前开发集上，S-learner 平均 Qini area 和 Top20 policy gain 最高；X-learner 的 cross-fitting 版本具有正向平均排序信号，但不是该统一学习器设定下的平均最优方法。seed2027 上条件 bootstrap 的 Qini 95% 区间为：T-learner `[0.000131, 0.000606]`，S-learner `[0.000244, 0.000669]`，X-learner `[0.000330, 0.000706]`。这些区间只表达固定验证集与固定模型分数下的估计不确定性。

固定 120,000 训练量的 X-learner Qini area 均值在 treated fraction 0.85、0.50、0.33、0.20 下分别为 `0.000259`、`0.000342`、`0.000312`、`0.000159`。当前结果没有呈现简单的单调关系，不能把某一比例解释为普遍最优。

## 6. 敏感性与限制

早期版本使用 `class_weight='balanced'`，会改变稀疏转化的隐含类别先验，导致 `predict_proba` 的概率校准发生变化；该版本仅作为敏感性参考，不能与未加权结果混合作 learner 结构结论。统一 HGB 主实验用于降低“模型结构”和“基础学习器”混杂，但仍未覆盖所有超参数与模型族。

当前开发集 validation 已用于模型探索，不能再冒充最终 holdout。Criteo 公开数据与真实业务人群、成本和收益分布不同；没有线上成本、收入和 A/B 实验时，离线 policy gain 不能写成 ROI 或线上收入提升。处理概率目前使用验证集 treatment 频率 plug-in，正式复现应核对官方设计概率。完整原始数据没有上传仓库。

## 7. 后续工作

保留未参与模型选择的最终 holdout，完成一次终局评估；统一记录数据版本和处理分配概率；补充同一基础学习器下的更多模型、bootstrap 方案和数据规模实验；在研究代码之外补一份结果表和图的自动化生成脚本。完成后可在简历中写为：

> 基于 Criteo 随机实验数据构建用户增量响应建模流程，统一比较 T/S/X-learner，并通过 IPW Qini、AUUC、Top-K policy gain 和固定训练量 treatment fraction 实验分析模型排序效果与稳健性。

## 8. 最终 holdout

模型配置冻结后，从原始全量数据中抽取与 1M 开发集按行号不重叠的 300,000 行 holdout。使用完整开发集训练 v2.1 配置，holdout 仅评估一次。Holdout Qini area 为：T-learner `0.000285`（95% 条件 bootstrap CI `[0.000067, 0.000501]`）、S-learner `0.000419`（`[0.000199, 0.000627]`）、X-learner + cross-fitting `0.000375`（`[0.000146, 0.000560]`）。S-learner 在该一次性 holdout 上仍为最高，但该比较只反映当前冻结配置与公开数据分布，不能外推线上收益。
