#!/usr/bin/env bash
# run_sweep.sh
# 用法：
#   ./run_sweep.sh run/<tag>/tasks.txt          # 全部执行
#   N=1 ./run_sweep.sh run/<tag>/tasks.txt      # 只跑前 N 条（测试）
#   INDEX=0 ./run_sweep.sh run/<tag>/tasks.txt  # 只跑第 INDEX 条（0-based）
#   DRY=1 ./run_sweep.sh run/<tag>/tasks.txt    # 只打印不执行

set -euo pipefail

TASKS="${1:-}"
if [ -z "$TASKS" ]; then
  echo "Usage: ./run_sweep.sh run/<tag>/tasks.txt"
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

: "${N:=}"
: "${INDEX:=}"
: "${DRY:=0}"

if [ ! -f "$TASKS" ]; then
  echo "[ERR] tasks file not found: $TASKS"
  exit 2
fi

total=$(grep -v '^[[:space:]]*$' "$TASKS" | wc -l | tr -d ' ')
echo "[INFO] total tasks = $total"

run_cmd() {
  local cmd="$1"
  echo "------------------------------------------------------------"
  echo "[CMD] $cmd"
  if [ "$DRY" -eq 1 ]; then
    echo "[DRY] skip"
    return 0
  fi
  # 每条命令里本身已经指定了 --out=...，
  # 但我们仍然建议把 stdout/stderr 重定向到 out_dir 内。
  # 这里简单做：先解析出 --out 后面的路径（粗略但够用）。
  out_dir=$(echo "$cmd" | sed -n 's/.*--out \([^ ]*\).*/\1/p')
  if [ -n "$out_dir" ]; then
    mkdir -p "$out_dir"
    bash -lc "$cmd" >"$out_dir/stdout.log" 2>"$out_dir/stderr.log"
  else
    bash -lc "$cmd"
  fi
}

# 跑指定 INDEX（0-based）
if [ -n "$INDEX" ]; then
  line=$((INDEX + 1))
  cmd=$(sed -n "${line}p" "$TASKS" || true)
  if [ -z "$cmd" ]; then
    echo "[ERR] INDEX out of range: $INDEX"
    exit 3
  fi
  run_cmd "$cmd"
  echo "[OK] done INDEX=$INDEX"
  exit 0
fi

# 跑前 N 个（不指定则全跑）
limit="$total"
if [ -n "$N" ]; then
  limit="$N"
fi
if [ "$limit" -gt "$total" ]; then
  limit="$total"
fi

i=0
while IFS= read -r cmd; do
  # 跳过空行
  [ -z "$(echo "$cmd" | tr -d '[:space:]')" ] && continue
  run_cmd "$cmd"
  i=$((i + 1))
  [ "$i" -ge "$limit" ] && break
done < "$TASKS"

echo "[OK] finished $i task(s)."
