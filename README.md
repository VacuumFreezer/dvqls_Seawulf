# Seawulf_simulation：Distributed VQLS Sweep/Slurm 

本目录用于在 SeaWulf（Slurm）上批量运行分布式量子电路优化（多问题 × 多优化器 × 多学习率 × 多拓扑）。

---

## 1. 目录结构（约定）

- `optimization/`
  - 各种优化脚本（同一套 CLI 参数）
  - 例：`met_line_tracking.py`, `met_line_tracking_AdamX.py`, `met_line_tracking_Adamz.py`, `met_line_tracking_GD.py`
- `problems/`
  - 不同方程/矩阵的 `static_ops` 定义（模块名不同）
  - 例：`static_ops_16agents_eq1_kappa196.py`, `static_ops_16agents_eq2_...py`, `static_ops_16agents_eq3_...py`
- `common/`
  - 通用函数模块（初始化、扁平化参数、tracking/consensus、reporting 等）
- `objective/`
  - builder / term 注入逻辑（例如 `incomplete_builder.py`）
- `params/`
  - sweep 配置文件，例如：`sweep.yaml`
- `run/`
  - 所有输出（每个任务一个 out_dir）
  - 推荐把 Slurm 外层日志放到 `run/<TAG>/slurm/`
- `make_tasks.py`
  - 读取 `params/sweep.yaml`，生成 `run/<TAG>/tasks.txt`
- `run_sweep.sh`
  - 本地/交互节点串行执行 `tasks.txt`（用于测试）
- `submit_array.slurm`
  - Slurm array 执行器（每个 array task 执行 tasks.txt 的一行命令）

---

## 2. 生成任务列表 tasks.txt（make_tasks.py）

- 生成任务（写入 run/<TAG>/tasks.txt，并创建各 out_dir）：

module load anaconda
conda activate pennylane

TAG=ep1000
python make_tasks.py --config params/sweep.yaml --tag $TAG


- 统计任务行数：

N=$(grep -cv '^[[:space:]]*$' run/$TAG/tasks.txt)
echo "Total tasks = $N"

---

## 3. 本地/登录节点串行测试（run_sweep.sh）

run_sweep.sh 读取 tasks.txt 串行执行（适合先测 1 条命令是否 OK）。

- 给执行权限：

chmod +x run_sweep.sh

- 只跑第 0 条（推荐）
INDEX=0 ./run_sweep.sh run/$TAG/tasks.txt

-  跑前 N 条
N=3 ./run_sweep.sh run/$TAG/tasks.txt

-  只打印不执行（检查命令）
DRY=1 ./run_sweep.sh run/$TAG/tasks.txt

---

## 4. Slurm 并行运行（方案B：Job Array，推荐）
- 先准备 Slurm 外层日志目录（避免日志堆积 & 避免目录不存在导致 FAIL）
mkdir -p run/$TAG/slurm

- 提交 array 作业（每个 array task 执行一条命令, 并发上限用 %K 控制（例如同时跑 8 个）：

sbatch --array=0-$((N-1))%8 --export=TAG=$TAG \
  -o run/$TAG/slurm/%x_%A_%a.out \
  -e run/$TAG/slurm/%x_%A_%a.err \
  submit_array.slurm

说明：

%A：ArrayJobId（主 job id）

%a：task id（0..N-1）

Slurm 的 .out/.err 会去 run/<TAG>/slurm/

每条 Python 命令的 stdout/stderr 仍会写到该任务自己的 --out 目录里的 stdout.log/stderr.log

## 5. 监控/排错命令

查看自己作业：

squeue --user=$USER


更清晰（含 pending reason）：

squeue --user=$USER -o "%.18i %.9P %.8j %.8T %.10M %R"

查看单个 task 细节（注意 array task 要带 _a）：

scontrol show job <JOBID>_<TASKID>

查看资源利用：

seff <JOBID>
sacct -j <JOBID> -l

取消作业：

取消整个 array：scancel <JOBID>

取消某一个 task：scancel <JOBID>_<TASKID>

## 6. 输出在哪里？

每个任务会在 --out 指定目录下产生：

report.txt（你 notebook 风格的报告）

metrics.jsonl（降频写入）

stdout.log / stderr.log（该任务 python 输出）

loss.png / diff.png（带 title：kappa≈cond(A), lr_g0/lr_a0, optimizer tag）

artifacts.npz（loss/diff history 等）

Slurm 外层日志在：run/<TAG>/slurm/