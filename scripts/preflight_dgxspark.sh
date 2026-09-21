#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/preflight_dgxspark.sh <head|worker> [deployment.env]

Read-only preflight for one DGX Spark node. It checks the local host against the
selected role in the deployment contract and never changes networking, clocks,
drivers, NCCL, or GPU state.
USAGE
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 64
fi

ROLE=$1
ENV_FILE=${2:-configs/dgxspark_tp2.env.example}
if [[ "$ROLE" != "head" && "$ROLE" != "worker" ]]; then
  echo "error: role must be head or worker" >&2
  exit 64
fi
if [[ ! -r "$ENV_FILE" ]]; then
  echo "error: cannot read deployment env: $ENV_FILE" >&2
  exit 2
fi

# The checked-in example is shell-compatible by design. For a private local
# contract, review it before sourcing; this script does not download env files.
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

failures=0
warnings=0
ok()   { printf 'OK    %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*"; warnings=$((warnings + 1)); }
fail() { printf 'FAIL  %s\n' "$*"; failures=$((failures + 1)); }

need_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    ok "command $1"
  else
    fail "missing command $1"
  fi
}

for cmd in ip nvidia-smi ethtool; do need_cmd "$cmd"; done

arch=$(uname -m)
if [[ "$arch" == "aarch64" || "$arch" == "arm64" ]]; then
  ok "architecture=$arch"
else
  fail "expected ARM64/aarch64, got $arch"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_line=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -n1 || true)
  if [[ "$gpu_line" == *"GB10"* ]]; then
    ok "GPU=$gpu_line"
  elif [[ -n "$gpu_line" ]]; then
    fail "expected NVIDIA GB10, got $gpu_line"
  else
    fail "nvidia-smi returned no GPU"
  fi
fi

if [[ "$ROLE" == "head" ]]; then
  expected_ip=${HEAD_IP:?HEAD_IP missing}
  net_if=${HEAD_CX7_IF:?HEAD_CX7_IF missing}
  ib_dev=${HEAD_CX7_IB:?HEAD_CX7_IB missing}
  gid_index=${HEAD_GID:-${NCCL_IB_GID_INDEX:?NCCL_IB_GID_INDEX missing}}
else
  expected_ip=${WORKER_IP:?WORKER_IP missing}
  net_if=${WORKER_CX7_IF:?WORKER_CX7_IF missing}
  ib_dev=${WORKER_CX7_IB:?WORKER_CX7_IB missing}
  gid_index=${WORKER_GID:-${NCCL_IB_GID_INDEX:?NCCL_IB_GID_INDEX missing}}
fi

if command -v ip >/dev/null 2>&1; then
  if ip -brief addr show "$net_if" >/tmp/ninfer_glm53_ip.$$ 2>/dev/null; then
    net_line=$(cat /tmp/ninfer_glm53_ip.$$)
    rm -f /tmp/ninfer_glm53_ip.$$
    if [[ "$net_line" == *"$expected_ip/"* ]]; then
      ok "$net_if owns $expected_ip"
    else
      fail "$net_if does not own expected IP $expected_ip ($net_line)"
    fi
  else
    rm -f /tmp/ninfer_glm53_ip.$$
    fail "interface $net_if does not exist"
  fi
fi

if command -v ethtool >/dev/null 2>&1 && [[ -e "/sys/class/net/$net_if" ]]; then
  speed=$(ethtool "$net_if" 2>/dev/null | awk -F': ' '/Speed:/{print $2; exit}')
  link=$(ethtool "$net_if" 2>/dev/null | awk -F': ' '/Link detected:/{print $2; exit}')
  if [[ "$speed" == "200000Mb/s" ]]; then
    ok "$net_if speed=$speed"
  else
    fail "$net_if expected 200000Mb/s, got ${speed:-unknown}"
  fi
  if [[ "$link" == "yes" ]]; then
    ok "$net_if link detected"
  else
    fail "$net_if link is ${link:-unknown}"
  fi
fi

gid_path="/sys/class/infiniband/$ib_dev/ports/1/gids/$gid_index"
if [[ -r "$gid_path" ]]; then
  gid=$(tr -d '[:space:]' < "$gid_path")
  if [[ -n "$gid" && "$gid" != "0000:0000:0000:0000:0000:0000:0000:0000" ]]; then
    ok "$ib_dev GID[$gid_index]=$gid"
  else
    fail "$ib_dev GID[$gid_index] is empty/zero"
  fi
else
  fail "missing RoCE GID path $gid_path"
fi

if command -v nvcc >/dev/null 2>&1; then
  cuda_release=$(nvcc --version | sed -n 's/.*release \([^,]*\).*/\1/p' | tail -n1)
  ok "nvcc release=${cuda_release:-unknown}"
else
  warn "nvcc not on PATH (runtime-only nodes may still be valid)"
fi

if [[ -r /proc/meminfo ]]; then
  mem_kib=$(awk '/MemTotal:/{print $2; exit}' /proc/meminfo)
  mem_gib=$((mem_kib / 1024 / 1024))
  if (( mem_gib >= 115 )); then
    ok "system unified memory ~= ${mem_gib} GiB"
  else
    warn "system memory ${mem_gib} GiB is below the expected ~121 GiB GB10 node"
  fi
fi

printf '\nsummary: failures=%d warnings=%d role=%s env=%s\n' "$failures" "$warnings" "$ROLE" "$ENV_FILE"
if (( failures > 0 )); then exit 2; fi
