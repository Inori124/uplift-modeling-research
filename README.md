# 用户增量响应建模：从随机实验到触达策略评估

这是一个可复现的小型研究项目，研究问题是：**在只有实验前用户特征的情况下，如何识别“被触达后更可能转化”的增量人群？不同 uplift 建模方法在处理组比例变化时是否稳定？**

> 当前仓库提供合成数据 smoke test 和完整研究框架。合成数据仅用于验证代码流程；接入 Criteo 公共数据后，才可填写正式实验结果。

## 研究设计

- 任务：估计个体处理效应（ITE），按预测 uplift 排序；
- 基线：T-learner（处理组／对照组分别训练逻辑回归）；
- 评估：独立测试集上的 IPW uplift 曲线、AUUC、Qini area、Top 10/20/30% uplift；
- 稳健性：只改变训练集处理组比例，测试集保持不变；0.5 和 0.25 表示处理组相对对照组的抽样比例；
- 防泄漏：只使用处理前特征，不使用 exposure、visit、conversion 等处理后变量；按用户一次性切分训练／测试。

AUUC/Qini 的计算采用测试集原始处理概率的 plug-in 估计 `p = mean(T)`。因此结果表示离线增量转化率排序效果，不能直接解释为线上 ROI。

## 运行

```bash
cd uplift-response-research
python3 src/run_experiment.py --out results/metrics.json
```

脚本只依赖 Python、NumPy 和 Pandas。完成 Criteo 数据接入后，将 `make_data` 替换为数据读取函数，并在报告中记录数据版本、样本量、特征定义和处理概率。

## 研究报告应包含

1. 数据与实验设计：随机分组、变量口径、处理前特征；
2. 方法：T/S/X-learner 或 uplift tree 的建模假设；
3. 指标：AUUC、Qini area、Top-K uplift 和置信区间；
4. 结果：主结果、处理组比例稳健性、不同 seed 重复实验；
5. 策略模拟：预算／触达成本假设、Top-K 策略、敏感性分析；
6. 限制：公开数据与线上业务分布不同，离线 uplift 不能替代线上 A/B 实验。

## 放进简历（完成实验后）

> 基于公开随机实验数据构建用户增量响应建模流程，使用处理前行为特征对比 T-learner、X-learner 与 uplift tree，并通过 Qini/AUUC 评估人群排序效果；固定独立测试集，改变训练集处理组比例开展有限样本稳健性实验，进一步完成预算约束下 Top-K 触达策略模拟。
