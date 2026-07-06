# SDCL / SYSU-MM01 项目开发总提示词

你正在接手 SDCL/SYSU-MM01 无监督跨模态行人重识别项目。项目路径：

/home/lhp/project/SDCL2

目标不是做小修小补，而是超过论文指标和当前复现最优。请先完整阅读本文件，再阅读仓库和日志，然后执行任务。

## 硬目标

核心指标优先级：

1. all-search Rank-1
2. all-search mAP
3. indoor mAP
4. indoor mINP

当前硬基准：

- 复现最优日志：`logs/0620/log复现最优.txt`
- best all-search: Rank-1 65.55 / mAP 63.41 / mINP 50.38
- best indoor: Rank-1 71.12 / mAP 76.58 / mINP 73.12

论文目标按当前记录：

- all-search mAP/mINP >= 63.24 / 51.06
- indoor mAP/mINP >= 76.90 / 73.50

不要只追求“略有提升”，目标是突破系统上限。

## 必读文件

请先阅读：

- `AGENTS.md`
- `SYSU_EXPERIMENTS.md`
- `logs/EXPERIMENT_NOTES.md`
- 当前主训练脚本，如 `sdcl_sysu_v*.py`
- 当前训练 sh，如 `train_sysu_v*.sh`
- `clustercontrast/`
- `tests/`
- 最近日志目录：`logs/0702`、`logs/0703`、`logs/0705`、`logs/v33`、`logs/V34`、`logs/v35`

如果发现本提示与仓库记录冲突，以仓库实际文件和最新日志为准，并说明冲突。

## 历史结论

不要继续无脑围绕 GPRD/邻域蒸馏小修小补。

历史 v15-v31/GPRD 基本没有超过复现最优。v36 CMBS hard pair 也失败，说明当前 CRA/shared pseudo label 不能直接当 hard positive 强拉。

当前应优先考虑：

- Stage-1 表征能力
- Stage-1 伪标签质量
- 跨模态聚类校正
- 样本选择
- 模型结构
- 训练稳定性
- Stage-1 到 Stage-2 的主闭环

不要把“换 checkpoint selector”当作真正大改。真正需要的是改 Stage-1 的模块、损失、记忆库、原型桥接或伪标签生成机制。

## 每次分析日志必须做

当我给你一个新日志，或者你自己发现新日志时，必须解析：

1. best model 指标
2. final epoch 指标
3. 单项最高 epoch 指标
4. 与论文指标比较
5. 与 `logs/0620/log复现最优.txt` 比较
6. 与上一版实验比较
7. 是否还在上升
8. 是否早停过早
9. Stage-2 是否退化
10. 聚类数量和 outlier 是否异常
11. 新增 loss 是否下降但指标不涨
12. 判断这次修改是有效、参数问题、实现问题、方向问题，还是实验噪声

必须输出表格：

| 实验名 | best all Rank-1/mAP/mINP | best indoor Rank-1/mAP/mINP | 相对复现最优差距 | 是否超过论文 |
| --- | --- | --- | --- | --- |

不要只说“可能需要调参”。必须给出下一步具体修改。

## 每次修改代码后必须做

每次修改后必须记录到：

`logs/EXPERIMENT_NOTES.md`

记录内容包括：

1. 新版本号，例如 v38
2. 修改目标
3. 修改文件
4. 核心逻辑
5. 预期改善哪个指标
6. 风险
7. 为什么比上一版可能更好
8. 验证结果
9. Ubuntu 训练命令

每次新版本修改时，把上一版主训练文件和对应 sh 放入 `main_try/`，避免主目录冗余。

## GitNexus 要求

AGENTS.md 要求：

- 修改符号前做 GitNexus impact
- 修改后做 detect_changes

如果 GitNexus 被历史 untracked / moved 文件干扰，要说明原因，但不要因此停止有效修改。

## 训练自动化要求

每次改完代码后，请直接给出并尽量启动训练命令。

训练建议使用 tmux 或 nohup，避免 Codex 会话断开导致训练中断。

推荐流程：

1. 创建日志目录
2. 启动训练
3. 记录 PID 或 tmux session
4. 每隔数小时检查：
   - 训练是否还在运行
   - log 是否新增
   - 是否出现 traceback / CUDA OOM / NaN
   - checkpoint 是否生成
   - best model 是否更新
5. 如果训练结束，立即解析日志，和复现最优/论文/上一版比较
6. 如果未结束，输出当前 epoch、当前最好指标、最近趋势，并继续等待或安排下次检查

如果需要，请创建脚本，例如：

- `scripts/run_experiment.sh`
- `scripts/watch_experiment.sh`
- `scripts/parse_sysu_log.py`

但不要用脚本替代判断。最终必须给出清晰结论。

## 初始化阅读与状态汇报

你现在第一步不要改代码。请先做：

1. `pwd`
2. `git status --short`
3. 阅读 `AGENTS.md`
4. 阅读 `SYSU_EXPERIMENTS.md`
5. 阅读 `logs/EXPERIMENT_NOTES.md`
6. 找出当前主目录有哪些 `sdcl_sysu_v*.py` 和 `train_sysu_v*.sh`
7. 找出最近日志目录和最新日志
8. 汇报：
   - 当前最新版本
   - 当前最好结果
   - 当前失败方向
   - 当前主训练文件
   - 下一步你建议的大改方向

初始化汇报后等待我确认，除非我明确说“按照计划修改”。