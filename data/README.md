# 本地数据说明

真实文件不提交到 GitHub。当前本地文件：`criteo-research-uplift-v2.1.csv`。

运行检查：

```bash
python3 src/inspect_criteo.py /path/to/criteo-research-uplift-v2.1.csv
```

字段：`f0`–`f11` 为特征，`treatment` 为处理分组，`conversion` 为转化结果，`visit` 和 `exposure` 为结果/处理相关变量。

建模时先只使用 `f0`–`f11`，不要使用 `visit`、`exposure` 或 `conversion` 作为特征。完整原始数据约 3.25 GB，不能提交到仓库。
