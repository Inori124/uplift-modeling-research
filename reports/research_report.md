# 用户增量响应建模研究（阶段性报告）

## 摘要

本研究使用 Criteo 公开随机实验数据，研究如何根据处理前用户特征识别触达带来的个体增量响应，并比较 T-learner、S-learner 与 X-learner。主实验在固定的 100 万行开发集上划分 75% train pool 与 25% validation，使用 5 个模型随机种子。主结果采用未加权 HistGradientBoostingClassifier 估计结果模型，X-learner 使用 3-fold training-only cross-fitting，所有模型的验证指标使用相同的 IPW 评估。

## 数据与变量

数据包含 13,979,592 行和 12 个匿名处理前特征 `f0`–`f11`。`treatment` 约为 0.85，`conversion` 约为 0.0029。`visit` 和 `exposure` 未作为模型特征。完整原始数据不提交仓库；本研究开发集由固定 seed=2027 的均匀抽样得到。

## 方法

T-learner 在 treatment/control 两组分别训练结果模型；S-learner 将 treatment 作为结果模型特征；X-learner 先以对侧结果模型构造伪处理效应，再分别训练处理效应模型，并按训练集 `p_train=mean(treatment)` 加权组合。X-learner 的 cross-fitting 用于减少结果模型过拟合造成的伪效应偏差，不表示解决所有数据泄漏。

## 评估

按预测 uplift 从高到低排序，使用测试集处理概率 plug-in `p=mean(treatment)` 计算 IPW 增量贡献。报告 AUUC、Qini area 和 Top-K policy gain。Top-K policy gain 的分母是整个验证集；若要表示被选中人群内部平均增量，需要除以相应选择比例。随机基线的理论 Qini area 为 0。

## 主结果

主结果见 `results/v2_summary.json`。结果仅属于固定开发集的验证估计，不是最终 holdout 泛化结论。当前结果表和图应与随机种子、模型版本和数据版本一起解读。

## 敏感性与稳健性

旧版实验比较了类别加权与未加权结果模型，也比较了处理组抽样比例；这些结果用于发现概率校准和样本量变化的影响，不作为 learner 结构的单独因果结论。类别加权会改变隐含类别先验，可能使 `predict_proba` 失去概率校准。处理组比例实验同时改变了处理比例和训练样本量，不能解释为纯 treatment ratio 效应。

## 限制与下一步

公开实验数据与真实业务分布不同；离线 Qini/AUUC 不能直接转化为线上 ROI。Criteo treatment 分配概率在本项目中以评估集 `mean(treatment)` 作 plug-in，需在最终报告中核对官方实验设计。当前模型组合仍存在学习器差异，主结果的统一 HGB 版本需要进一步补充固定训练总量的 treatment fraction 实验。下一步应保留未参与模型选择的最终 holdout，完成统一学习器下的比例稳健性、更多 bootstrap 重复和最终一次性评估。
