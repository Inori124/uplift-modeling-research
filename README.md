# Uplift Modeling Research

一个使用 Criteo 随机实验数据研究用户增量响应建模的可复现实验项目。

## 项目背景

互联网公司会通过广告、Push、优惠券或站内推荐触达用户，希望用户完成访问、注册或购买。传统转化模型回答“哪些用户最可能转化”，增长策略更关心“哪些用户是因为被触达后才增加转化”。本来就会购买的用户不一定值得消耗触达成本；真正有价值的是接受处理后才增加响应的人群。

本项目研究如何利用随机 treatment/control 实验数据识别增量响应人群，并在有限触达预算下对用户进行排序。对用户特征 `X`、处理分组 `T` 和转化结果 `Y`，目标是估计条件平均处理效应（CATE）：

```text
τ(x) = P(Y=1 | T=1, X=x) - P(Y=1 | T=0, X=x)
```

Criteo 数据包含处理前匿名特征和实验后的转化结果。项目只使用 `f0`–`f11` 作为特征，`treatment` 作为处理变量，`conversion` 作为结果变量；`visit`、`exposure` 不进入模型，避免把处理后的信息带入预测。

## 数据

真实数据文件约 3.25 GB，不能上传 GitHub。下载后放在本地 `data/` 目录，运行：

```bash
python3 src/inspect_criteo.py data/criteo-research-uplift-v2.1.csv
python3 src/sample_data.py data/criteo-research-uplift-v2.1.csv --out data/criteo-dev-1m.csv --n 1000000 --seed 2027
python3 src/baseline_stats.py data/criteo-dev-1m.csv
```

当前数据检查：全量 13,979,592 行；`treatment` 比例约 0.85；`conversion` 比例约 0.0029；字段无缺失。完整字段说明见 [`data/README.md`](data/README.md)。

## 研究设计

v2 主实验固定 `split_seed=2027`，将 100 万行开发集划分为 75% train pool 和 25% validation，并使用 5 个模型随机种子（2027–2031）。所有模型使用相同的未加权 `HistGradientBoostingClassifier` 结果模型；X-learner 的两组效应模型使用 `HistGradientBoostingRegressor`，其 nuisance outcome 预测使用 3-fold training-only cross-fitting。

比较方法：

- Random draw：随机排序，仅作有噪声的随机排序参考；理论随机期望在 `evaluation_v2.py` 中单独计算；
- T-learner：处理组和对照组分别训练结果模型；
- S-learner：将 `treatment` 作为结果模型特征；
- X-learner：用对侧结果模型构造伪处理效应，再训练两组效应模型。

评估按预测 uplift 排序，使用验证集 treatment 比例的 plug-in IPW 估计，报告 AUUC、Qini area 和 Top-K policy gain。`policy_gain` 的分母是整个评估集；被选中人群内部的平均增量另报告为 `topK_selected_ate`。这些是离线增量转化估计，不能直接写成线上 ROI。

## v2 主结果

| 方法 | Qini area 均值 ± 标准差 | Top20 policy gain 均值 ± 标准差 |
| --- | ---: | ---: |
| Random draw | 0.000005 ± 0.000047 | 0.000274 ± 0.000115 |
| T-learner HGB | 0.000412 ± 0.000037 | 0.001080 ± 0.000048 |
| S-learner HGB | 0.000518 ± 0.000087 | 0.001205 ± 0.000118 |
| X-learner HGB + cross-fitting | 0.000427 ± 0.000070 | 0.001060 ± 0.000066 |

在当前固定开发集上，S-learner 的平均 Qini area 和 Top20 policy gain 最高；X-learner 不是本统一 HGB 设定下的平均最优方法。该结论只适用于当前开发集验证；最终 holdout 结果见下文。

## 固定训练量稳健性

固定训练样本量为 120,000，只改变 treated fraction，测试集仍为同一份 validation；每个设定使用 5 个 seed。X-learner Qini area 均值如下：

| Treated fraction | Qini area 均值 | 标准差 |
| ---: | ---: | ---: |
| 0.85 | 0.000259 | 0.000198 |
| 0.50 | 0.000342 | 0.000045 |
| 0.33 | 0.000312 | 0.000068 |
| 0.20 | 0.000159 | 0.000111 |

该实验只说明当前固定样本量与训练构成设定下的离线变化，不是线上 treatment policy 的因果效应。

## 一键复现实验

```bash
PYTHONPATH=src python3 src/run_study_v2.py --suite all --workers 2 --bootstrap 200
```

该命令需要本地准备开发集和原始数据；运行时间取决于机器。

## 结果、报告与测试

- [研究报告](reports/research_report.md)
- [v2 汇总结果](results/v2_summary.json)
- [严格 tie-safe 指标与 paired bootstrap](src/evaluation_v2.py)
- [v2 Qini 曲线](reports/figures/v2_qini_curves_seed2027.png)
- [处理组构成稳健性图](reports/figures/v2_fixed_size_treatment_robustness.png)
- [论文数据概览图](reports/figures/paper_data_overview.png)
- [论文开发集 Qini 曲线](reports/figures/paper_qini_validation.png)
- [论文 holdout Qini 曲线](reports/figures/paper_qini_holdout.png)
- [论文 holdout 区间图](reports/figures/paper_holdout_qini_ci.png)
- [论文 treatment fraction 图](reports/figures/paper_treatment_fraction.png)
- [类别加权敏感性图（legacy）](reports/figures/class_weight_sensitivity.png)

```bash
python3 -m unittest discover -s tests -v
MPLBACKEND=Agg python3 src/plot_v2.py
```

Bootstrap 区间是固定已训练模型分数、按 treatment 分层重抽样的条件区间；它不包含重新训练和模型选择不确定性。`data/` 下原始 CSV 和开发集均已忽略，不会提交到仓库。

## 后续工作

1. 补充跨时间或外部数据验证；
2. 检查官方 treatment assignment probability，并以设计概率替代 plug-in；
3. 增加 bootstrap 重复次数，比较概率校准方法；
4. 完成后再把真实结果写入简历，表述为“基于公开随机实验数据的个人研究项目”。

## 最终 holdout（一次性评估）

在模型配置和研究问题冻结后，从原始数据中另取 300,000 行，与 1M 开发集按行号不重叠。使用完整开发集训练同一 v2.1 配置，只在该 holdout 上评估一次；holdout 没有参与模型选择。结果文件为 [`results/final_holdout.json`](results/final_holdout.json)，生成脚本为 [`src/run_final_holdout.py`](src/run_final_holdout.py)。

| 方法 | Holdout Qini area | Top20 policy gain | Qini 95% 条件 bootstrap CI |
| --- | ---: | ---: | ---: |
| Random draw | 0.000031 | 0.000221 | [-0.000111, 0.000153] |
| T-learner HGB | 0.000285 | 0.000732 | [0.000067, 0.000501] |
| S-learner HGB | 0.000419 | 0.000887 | [0.000199, 0.000627] |
| X-learner HGB + cross-fitting | 0.000375 | 0.000838 | [0.000146, 0.000560] |

Holdout 结果与开发集方向一致：在冻结的统一 HGB 配置下，S-learner 的 Qini area 最高，X-learner 次之。该结论仍是公开随机实验数据上的离线结果，不代表线上 ROI 或业务收益。
