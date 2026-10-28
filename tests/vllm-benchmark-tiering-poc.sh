#!/usr/bin/env bash
set -Eeuo pipefail

# KVShrink Tiered Storage PoC Benchmark Runner (G3.5 Scenario 1 & Scenario 2).
#
# Covers three comparison cases (without explicit prefetch):
#   Case A: Baseline — vLLM without KVShrink external KV cache reuse
#   Case B: DDR-Only (Bounded) — KVShrink enabled, DDR pool capacity bounded,
#           background watermark eviction enabled WITHOUT persistence
#           (IAXL_CACHE_AUTO_SPILL_ENABLE=1, IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=1)
#   Case C: DDR + Storage Tiering — KVShrink enabled, DDR pool capacity bounded,
#           background watermark Persist + Evict enabled
#           (IAXL_CACHE_AUTO_SPILL_ENABLE=1, IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=0)
#
# Supports both:
#   - G3.5 Scenario 1 (Local HeadNode KVStore + JBOF/NVMe-oF mount, IAXL_RDMA_ENABLE=0)
#   - G3.5 Scenario 2 (Cross-Node GDR remote_pool Daemon + local NVMe, IAXL_RDMA_ENABLE=1)
#
# Usage examples:
#   1) Run against currently running server (auto-detects Case A, B, or C):
#      MODEL=/path/to/model bash tests/vllm-benchmark-tiering-poc.sh
#
#   2) Run full A/B/C sweep with start/stop hooks or manual confirmation:
#      CASES="A B C" SERVER_LIFECYCLE_MODE=restart \
#      MODEL=/path/to/model bash tests/vllm-benchmark-tiering-poc.sh

MODEL=${MODEL:?"MODEL must be set"}
HOST=${HOST:-localhost}
PORT=${PORT:-8000}

_daemon_ctrl_ip="${IAXL_RDMA_DAEMON_NIC_IPS%%,*}"
REMOTE_CACHE_HOST="${REMOTE_CACHE_HOST:-${_daemon_ctrl_ip:-${IAXL_RDMA_DAEMON_IP:-localhost}}}"
REMOTE_CACHE_PORT="${REMOTE_CACHE_PORT:-${IAXL_API_CONTROLLER_PORT:-18700}}"
METRICS_WORKER_BASE_PORT="${METRICS_WORKER_BASE_PORT:-${IAXL_API_WORKER_BASE_PORT:-18800}}"
METRICS_TP_SIZE="${METRICS_TP_SIZE:-${TP_SIZE:-${IAXL_RDMA_TP_SIZE:-4}}}"
API_BASE=${API_BASE:-http://${REMOTE_CACHE_HOST}:${REMOTE_CACHE_PORT}}

CASES_WAS_SET=0
if [[ -v CASES ]]; then
  CASES_WAS_SET=1
fi
CASES=${CASES:-"A B C"}
START_CASE_CMD=${START_CASE_CMD:-}
STOP_CASE_CMD=${STOP_CASE_CMD:-}
RESET_BACKEND_CMD=${RESET_BACKEND_CMD:-}
MANUAL_CASE_CONFIRM=${MANUAL_CASE_CONFIRM:-1}
WAIT_FOR_SERVER_SECONDS=${WAIT_FOR_SERVER_SECONDS:-180}
SERVER_CHECK_INTERVAL_SECONDS=${SERVER_CHECK_INTERVAL_SECONDS:-2}
SERVER_LIFECYCLE_MODE=${SERVER_LIFECYCLE_MODE:-auto}
BENCH_RUNNER=${BENCH_RUNNER:-auto}
BENCH_CONTAINER_NAME=${BENCH_CONTAINER_NAME:-iaxl.vllm}

# Service-side IAXL tiering variables recorded and exported for hooks.
IAXL_DDR_POOL_SIZE_GB=${IAXL_DDR_POOL_SIZE_GB:-8}
IAXL_CACHE_DIR=${IAXL_CACHE_DIR:-_data/kvcache}
IAXL_KV_COMPRESSION=${IAXL_KV_COMPRESSION:-1}
IAXL_CACHE_AUTO_SPILL_ENABLE=${IAXL_CACHE_AUTO_SPILL_ENABLE:-1}
IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=${IAXL_CACHE_AUTO_SPILL_EVICT_ONLY:-0}
IAXL_CACHE_HIGH_WATERMARK=${IAXL_CACHE_HIGH_WATERMARK:-0.80}
IAXL_CACHE_LOW_WATERMARK=${IAXL_CACHE_LOW_WATERMARK:-0.60}
IAXL_CACHE_SPILL_BATCH_GROUPS=${IAXL_CACHE_SPILL_BATCH_GROUPS:-16}
IAXL_CACHE_SPILL_PACE_MS=${IAXL_CACHE_SPILL_PACE_MS:-2}
IAXL_CACHE_EVICT_GRACE_MS=${IAXL_CACHE_EVICT_GRACE_MS:-50}

# Case-specific descriptors and exported hints for START_CASE_CMD hooks.
IAXL_CASE_A_DESC=${IAXL_CASE_A_DESC:-baseline_no_kv_reuse}
IAXL_CASE_B_DESC=${IAXL_CASE_B_DESC:-kvshrink_ddr_bounded_auto_evict_only}
IAXL_CASE_C_DESC=${IAXL_CASE_C_DESC:-kvshrink_ddr_storage_auto_persist_and_evict}

# Benchmark workload configuration.
# BENCH_WORKLOAD_MODE=revisit:
#   Stage 1 (warm_target): populate target prefix set (seed)
#   Stage 2 (pressure):    run pressure prefix set (pressure_seed) exceeding DDR pool capacity
#                          so background worker evicts (Case B) or persists+evicts (Case C) Stage 1
#   Stage 3 (measure):     revisit target prefix set (seed) and measure TTFT & throughput
BENCH_WORKLOAD_MODE=${BENCH_WORKLOAD_MODE:-revisit}
INPUT_LENS=${INPUT_LENS:-"2000 4000 8000"}
HIT_RATES=${HIT_RATES:-80}
BENCH_CONFIGS_1000=${BENCH_CONFIGS_1000:-"16 32 64"}
BENCH_CONFIGS_2000=${BENCH_CONFIGS_2000:-"16 32 64"}
BENCH_CONFIGS_4000=${BENCH_CONFIGS_4000:-"16 32"}
BENCH_CONFIGS_8000=${BENCH_CONFIGS_8000:-"8 16"}
BENCH_CONFIGS_16000=${BENCH_CONFIGS_16000:-"4 8"}
BENCH_CONFIGS_DEFAULT=${BENCH_CONFIGS_DEFAULT:-"16"}

OUTPUT_LEN=${OUTPUT_LEN:-128}
NUM_WARMUPS=${NUM_WARMUPS:-5}
WARM_NUM_PROMPTS=${WARM_NUM_PROMPTS:-100}
MEASURE_NUM_PROMPTS=${MEASURE_NUM_PROMPTS:-200}
REVISIT_WARM_NUM_PROMPTS=${REVISIT_WARM_NUM_PROMPTS:-$WARM_NUM_PROMPTS}
REVISIT_MEASURE_NUM_PROMPTS=${REVISIT_MEASURE_NUM_PROMPTS:-$REVISIT_WARM_NUM_PROMPTS}
PRESSURE_NUM_PROMPTS=${PRESSURE_NUM_PROMPTS:-$MEASURE_NUM_PROMPTS}
PRESSURE_SEED_OFFSET=${PRESSURE_SEED_OFFSET:-1000000}
RUN_WARM_FOR_BASELINE=${RUN_WARM_FOR_BASELINE:-0}
BASE_SEED=${BASE_SEED:-10000}
MEASURE_SEED_OFFSET=${MEASURE_SEED_OFFSET:-0}
REQUEST_RATE=${REQUEST_RATE:-inf}
VLLM_BENCH_EXTRA_ARGS=${VLLM_BENCH_EXTRA_ARGS:-}

# Drain & synchronization controls.
PRE_CLEANUP_SLEEP=${PRE_CLEANUP_SLEEP:-5}
POST_CLEANUP_SLEEP=${POST_CLEANUP_SLEEP:-3}
STATUS_SAMPLE_INTERVAL_SECONDS=${STATUS_SAMPLE_INTERVAL_SECONDS:-2}
STATUS_CURL_TIMEOUT_SECONDS=${STATUS_CURL_TIMEOUT_SECONDS:-10}
WAIT_SPILL_SECONDS=${WAIT_SPILL_SECONDS:-300}
SPILL_BEFORE_REVISIT=${SPILL_BEFORE_REVISIT:-0}
EVICT_COUNT=${EVICT_COUNT:-500000000000}
EVICT_BETWEEN_CASES=${EVICT_BETWEEN_CASES:-1}
EVICT_BEFORE_EACH_MEASURE=${EVICT_BEFORE_EACH_MEASURE:-1}
EVICT_AFTER_EACH_MEASURE=${EVICT_AFTER_EACH_MEASURE:-1}

# PoC Evaluation criteria (Case C vs Case A & Case B).
P95_IMPROVEMENT_TARGET_PCT=${P95_IMPROVEMENT_TARGET_PCT:-15}
P99_IMPROVEMENT_TARGET_PCT=${P99_IMPROVEMENT_TARGET_PCT:-20}
THROUGHPUT_DROP_TOLERANCE_PCT=${THROUGHPUT_DROP_TOLERANCE_PCT:-0}
REQUIRE_STORAGE_HITS=${REQUIRE_STORAGE_HITS:-1}

LOG_DIR=${LOG_DIR:-logs_tiering_poc_$(date +'%Y%m%d_%H%M%S')}
SUMMARY="$LOG_DIR/benchmark_summary.csv"
COMPARE="$LOG_DIR/case_comparison.csv"
REPORT="$LOG_DIR/conclusion.md"
META="$LOG_DIR/run_metadata.env"
mkdir -p "$LOG_DIR"

LIVE_SERVER_VLLM_READY=0
LIVE_SERVER_KVSTORE_READY=0
LIVE_SERVER_AUTO_SPILL_ENABLED=0
LIVE_SERVER_EVICT_ONLY=0
LIVE_SERVER_CASE_HINT=""
BENCH_RUNNER_CMD=()

case_desc() {
  case "$1" in
    A) printf "%s" "$IAXL_CASE_A_DESC" ;;
    B) printf "%s" "$IAXL_CASE_B_DESC" ;;
    C) printf "%s" "$IAXL_CASE_C_DESC" ;;
    *) printf "custom_%s" "$1" ;;
  esac
}

case_kvshrink_enable() {
  case "$1" in
    A) printf "0" ;;
    B|C) printf "1" ;;
    *) printf "1" ;;
  esac
}

case_auto_spill_enable() {
  case "$1" in
    A) printf "0" ;;
    B|C) printf "1" ;;
    *) printf "%s" "$IAXL_CACHE_AUTO_SPILL_ENABLE" ;;
  esac
}

case_auto_spill_evict_only() {
  case "$1" in
    A) printf "0" ;;
    B) printf "1" ;;
    C) printf "0" ;;
    *) printf "%s" "$IAXL_CACHE_AUTO_SPILL_EVICT_ONLY" ;;
  esac
}

concurrencies_for_len() {
  local input_len=$1
  local var_name="BENCH_CONFIGS_${input_len}"
  local value=${!var_name:-}
  if [[ -n "$value" ]]; then
    printf "%s" "$value"
  else
    printf "%s" "$BENCH_CONFIGS_DEFAULT"
  fi
}

api_status() {
  local out=$1
  if ! curl --noproxy '*' -fsS --max-time "$STATUS_CURL_TIMEOUT_SECONDS" "$API_BASE/v1/cache/status" > "$out" 2>/dev/null; then
    printf '{}\n' > "$out"
  fi
}

api_post() {
  local path=$1
  local body=$2
  local out=$3
  if ! curl --noproxy '*' -fsS --max-time 120 -X POST "$API_BASE$path" \
      -H 'Content-Type: application/json' -d "$body" > "$out" 2>/dev/null; then
    printf '{}\n' > "$out"
  fi
}

reset_case_metrics() {
  if curl --noproxy '*' --fail --silent --show-error --get \
       "http://${REMOTE_CACHE_HOST}:${REMOTE_CACHE_PORT}/v1/cache/metrics" \
       --data-urlencode "enable=1" \
       --data-urlencode "reset=1" > /dev/null 2>&1; then
    return 0
  fi
  local rank
  for ((rank = 0; rank < METRICS_TP_SIZE; rank++)); do
    curl --noproxy '*' --fail --silent --show-error --get \
      "http://${REMOTE_CACHE_HOST}:$((METRICS_WORKER_BASE_PORT + rank))/v1/cache/metrics" \
      --data-urlencode "enable=1" \
      --data-urlencode "reset=1" > /dev/null 2>&1 || true
  done
}

collect_case_metrics() {
  local metrics_file=$1
  if curl --noproxy '*' --fail --silent --show-error --get \
       "http://${REMOTE_CACHE_HOST}:${REMOTE_CACHE_PORT}/v1/cache/metrics" \
       > "$metrics_file" 2>/dev/null; then
    return 0
  fi
  printf '{}\n' > "$metrics_file"
}

json_scalar() {
  local file=$1
  local field=$2
  local mode=${3:-sum}
  python3 - "$file" "$field" "$mode" <<'PY'
import json
import sys
path, field, mode = sys.argv[1:]
try:
    data = json.load(open(path, "r", encoding="utf-8"))
except Exception:
    print("")
    raise SystemExit
workers = data.get("workers") if isinstance(data, dict) else None
if isinstance(workers, dict):
    items = [v for v in workers.values() if isinstance(v, dict) and "error" not in v]
elif isinstance(data, dict):
    items = [data]
else:
    items = []

def nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

values = [nested(item, field) for item in items]
values = [v for v in values if v is not None]
nums = [float(v) for v in values if isinstance(v, (int, float, bool))]
if nums:
    if mode == "first":
        print(nums[0])
    elif mode == "max":
        print(max(nums))
    else:
        print(sum(nums))
elif values:
    print(values[0])
else:
    print("")
PY
}

fetch_live_server_state() {
  LIVE_SERVER_VLLM_READY=0
  LIVE_SERVER_KVSTORE_READY=0
  LIVE_SERVER_AUTO_SPILL_ENABLED=0
  LIVE_SERVER_EVICT_ONLY=0
  LIVE_SERVER_CASE_HINT=""

  if curl --noproxy '*' -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
    LIVE_SERVER_VLLM_READY=1
  fi

  local status_file
  status_file=$(mktemp)
  api_status "$status_file"
  if grep -q '"workers"' "$status_file"; then
    LIVE_SERVER_KVSTORE_READY=1
    local auto_spill_enabled evict_only
    auto_spill_enabled=$(json_scalar "$status_file" "auto_spill.enabled" "max")
    evict_only=$(json_scalar "$status_file" "auto_spill.evict_only" "max")
    if [[ "$auto_spill_enabled" == "1.0" || "$auto_spill_enabled" == "1" || "$auto_spill_enabled" == "true" || "$auto_spill_enabled" == "True" ]]; then
      LIVE_SERVER_AUTO_SPILL_ENABLED=1
    fi
    if [[ "$evict_only" == "1.0" || "$evict_only" == "1" || "$evict_only" == "true" || "$evict_only" == "True" ]]; then
      LIVE_SERVER_EVICT_ONLY=1
    fi
  fi
  rm -f "$status_file"

  if [[ "$LIVE_SERVER_VLLM_READY" != "1" ]]; then
    LIVE_SERVER_CASE_HINT=""
  elif [[ "$LIVE_SERVER_KVSTORE_READY" != "1" ]]; then
    LIVE_SERVER_CASE_HINT="A"
  elif [[ "$LIVE_SERVER_AUTO_SPILL_ENABLED" == "1" && "$LIVE_SERVER_EVICT_ONLY" == "1" ]]; then
    LIVE_SERVER_CASE_HINT="B"
  elif [[ "$LIVE_SERVER_AUTO_SPILL_ENABLED" == "1" && "$LIVE_SERVER_EVICT_ONLY" == "0" ]]; then
    LIVE_SERVER_CASE_HINT="C"
  else
    LIVE_SERVER_CASE_HINT="B"
  fi
}

case_supported_by_live_server() {
  local case_id=$1
  case "$case_id" in
    A)
      [[ "$LIVE_SERVER_VLLM_READY" == "1" && "$LIVE_SERVER_KVSTORE_READY" != "1" ]]
      ;;
    B)
      [[ "$LIVE_SERVER_VLLM_READY" == "1" && "$LIVE_SERVER_KVSTORE_READY" == "1" && ( "$LIVE_SERVER_EVICT_ONLY" == "1" || "$LIVE_SERVER_AUTO_SPILL_ENABLED" != "1" ) ]]
      ;;
    C)
      [[ "$LIVE_SERVER_VLLM_READY" == "1" && "$LIVE_SERVER_KVSTORE_READY" == "1" && "$LIVE_SERVER_AUTO_SPILL_ENABLED" == "1" && "$LIVE_SERVER_EVICT_ONLY" == "0" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

resolve_server_lifecycle_mode() {
  fetch_live_server_state

  case "$SERVER_LIFECYCLE_MODE" in
    auto)
      if [[ -n "$START_CASE_CMD" || -n "$STOP_CASE_CMD" || -n "$RESET_BACKEND_CMD" ]]; then
        SERVER_LIFECYCLE_MODE=restart
      elif [[ "$LIVE_SERVER_VLLM_READY" == "1" ]]; then
        SERVER_LIFECYCLE_MODE=current
      else
        SERVER_LIFECYCLE_MODE=restart
      fi
      ;;
    current|restart)
      ;;
    *)
      echo "ERROR: unsupported SERVER_LIFECYCLE_MODE=$SERVER_LIFECYCLE_MODE (use auto/current/restart)" >&2
      exit 1
      ;;
  esac

  if [[ "$SERVER_LIFECYCLE_MODE" == "current" ]]; then
    if [[ "$LIVE_SERVER_VLLM_READY" != "1" ]]; then
      echo "ERROR: SERVER_LIFECYCLE_MODE=current requires a running vLLM server at http://${HOST}:${PORT}" >&2
      exit 1
    fi

    if [[ "$CASES_WAS_SET" != "1" ]]; then
      case "$LIVE_SERVER_CASE_HINT" in
        A|B|C)
          CASES="$LIVE_SERVER_CASE_HINT"
          ;;
        *)
          echo "ERROR: unable to infer benchmark CASES from the running server; set CASES explicitly or use SERVER_LIFECYCLE_MODE=restart" >&2
          exit 1
          ;;
      esac
    fi

    local case_id
    for case_id in $CASES; do
      if ! case_supported_by_live_server "$case_id"; then
        echo "ERROR: requested case $case_id does not match the running server configuration (detected live case hint: ${LIVE_SERVER_CASE_HINT:-unknown})" >&2
        echo "       Use SERVER_LIFECYCLE_MODE=restart with START_CASE_CMD/STOP_CASE_CMD, or adjust CASES." >&2
        exit 1
      fi
    done
  fi
}

resolve_bench_runner() {
  case "$BENCH_RUNNER" in
    auto)
      if command -v vllm >/dev/null 2>&1; then
        BENCH_RUNNER=local
      elif docker ps --format '{{.Names}}' | grep -Fxq "$BENCH_CONTAINER_NAME"; then
        BENCH_RUNNER=container
      else
        echo "ERROR: neither local 'vllm' nor running container '$BENCH_CONTAINER_NAME' is available for benchmark execution" >&2
        exit 1
      fi
      ;;
    local)
      if ! command -v vllm >/dev/null 2>&1; then
        echo "ERROR: BENCH_RUNNER=local but 'vllm' is not installed on the host" >&2
        exit 1
      fi
      ;;
    container)
      if ! docker ps --format '{{.Names}}' | grep -Fxq "$BENCH_CONTAINER_NAME"; then
        echo "ERROR: BENCH_RUNNER=container but container '$BENCH_CONTAINER_NAME' is not running" >&2
        exit 1
      fi
      ;;
    *)
      echo "ERROR: unsupported BENCH_RUNNER=$BENCH_RUNNER (use auto/local/container)" >&2
      exit 1
      ;;
  esac

  if [[ "$BENCH_RUNNER" == "local" ]]; then
    BENCH_RUNNER_CMD=(vllm bench serve)
  else
    BENCH_RUNNER_CMD=(docker exec "$BENCH_CONTAINER_NAME" vllm bench serve)
  fi
}

wait_for_server() {
  local deadline=$((SECONDS + WAIT_FOR_SERVER_SECONDS))
  while (( SECONDS < deadline )); do
    if curl --noproxy '*' -fsS --max-time 2 "http://${HOST}:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$SERVER_CHECK_INTERVAL_SECONDS"
  done
  echo "ERROR: vLLM server is not ready at http://${HOST}:${PORT}" >&2
  return 1
}

monitor_status() {
  local out=$1
  while true; do
    local tmp
    tmp=$(mktemp)
    api_status "$tmp"
    python3 - "$tmp" <<'PY' >> "$out"
import json
import sys
import time
try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    data = {}
print(json.dumps({"ts": time.time(), "status": data}, separators=(",", ":")))
PY
    rm -f "$tmp"
    sleep "$STATUS_SAMPLE_INTERVAL_SECONDS"
  done
}

extract_metric() {
  local pattern=$1
  local log=$2
  grep "$pattern" "$log" | tail -n 1 | awk '{print $NF}' || true
}

run_bench() {
  local log=$1
  local input_len=$2
  local hit_rate=$3
  local concurrency=$4
  local seed=$5
  local num_prompts=$6
  local random_prefix_len=$((input_len * hit_rate / 100))
  local random_input_len=$((input_len - random_prefix_len))

  # shellcheck disable=SC2086
  "${BENCH_RUNNER_CMD[@]}" \
    --backend vllm \
    --model "$MODEL" --tokenizer "$MODEL" \
    --dataset-name random \
    --host "$HOST" --port "$PORT" \
    --random-input-len "$random_input_len" \
    --random-prefix-len "$random_prefix_len" \
    --random-output-len "$OUTPUT_LEN" \
    --ignore-eos \
    --percentile-metrics "ttft,tpot" \
    --metric-percentiles "50,90,95,99" \
    --seed "$seed" \
    --trust-remote-code \
    --request-rate "$REQUEST_RATE" \
    --num-warmups "$NUM_WARMUPS" \
    --num-prompts "$num_prompts" \
    --max-concurrency "$concurrency" \
    $VLLM_BENCH_EXTRA_ARGS \
    > "$log" 2>&1
}

evict_cache() {
  local out=$1
  sleep "$PRE_CLEANUP_SLEEP"
  api_post "/v1/cache/evict" "{\"count\": ${EVICT_COUNT}}" "$out"
  sleep "$POST_CLEANUP_SLEEP"
}

spill_cache() {
  local out=$1
  local evict_only=${2:-false}
  api_post "/v1/cache/spill" "{\"target_usage\": ${IAXL_CACHE_LOW_WATERMARK}, \"evict_only\": ${evict_only}}" "$out"
}

wait_flag_false() {
  local dir=$1
  local field=$2
  local timeout=$3
  local deadline=$((SECONDS + timeout))
  while (( SECONDS < deadline )); do
    local status_file="$dir/status_wait_${field//./_}_${SECONDS}.json"
    api_status "$status_file"
    local running
    running=$(json_scalar "$status_file" "$field" "max")
    if [[ "$running" != "1.0" && "$running" != "1" && "$running" != "true" && "$running" != "True" ]]; then
      return 0
    fi
    sleep "$STATUS_SAMPLE_INTERVAL_SECONDS"
  done
  echo "WARN: timeout waiting for $field=false" >&2
}

prepare_case() {
  local case_id=$1
  export CASE_ID=$case_id
  export CASE_DESC
  CASE_DESC=$(case_desc "$case_id")
  export CASE_KVSHRINK_ENABLE
  CASE_KVSHRINK_ENABLE=$(case_kvshrink_enable "$case_id")
  export IAXL_CACHE_AUTO_SPILL_ENABLE
  IAXL_CACHE_AUTO_SPILL_ENABLE=$(case_auto_spill_enable "$case_id")
  export IAXL_CACHE_AUTO_SPILL_EVICT_ONLY
  IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=$(case_auto_spill_evict_only "$case_id")
  export IAXL_DDR_POOL_SIZE_GB IAXL_CACHE_DIR IAXL_KV_COMPRESSION
  export IAXL_CACHE_HIGH_WATERMARK IAXL_CACHE_LOW_WATERMARK
  export IAXL_CACHE_SPILL_BATCH_GROUPS IAXL_CACHE_SPILL_PACE_MS IAXL_CACHE_EVICT_GRACE_MS

  echo "================================================================"
  echo "Case ${CASE_ID}: ${CASE_DESC}"
  echo "  CASE_KVSHRINK_ENABLE=${CASE_KVSHRINK_ENABLE}"
  echo "  IAXL_DDR_POOL_SIZE_GB=${IAXL_DDR_POOL_SIZE_GB}"
  echo "  IAXL_CACHE_DIR=${IAXL_CACHE_DIR}"
  echo "  IAXL_CACHE_AUTO_SPILL_ENABLE=${IAXL_CACHE_AUTO_SPILL_ENABLE}"
  echo "  IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=${IAXL_CACHE_AUTO_SPILL_EVICT_ONLY}"
  echo "  IAXL_CACHE_HIGH_WATERMARK=${IAXL_CACHE_HIGH_WATERMARK}"
  echo "  IAXL_CACHE_LOW_WATERMARK=${IAXL_CACHE_LOW_WATERMARK}"
  echo "  REMOTE_CACHE_ENDPOINT=${API_BASE}"
  echo "  SERVER_LIFECYCLE_MODE=${SERVER_LIFECYCLE_MODE}"
  echo "================================================================"

  if [[ "$SERVER_LIFECYCLE_MODE" == "current" ]]; then
    fetch_live_server_state
    if ! case_supported_by_live_server "$case_id"; then
      echo "ERROR: live server no longer matches case ${CASE_ID}; restart the service or switch to SERVER_LIFECYCLE_MODE=restart" >&2
      return 1
    fi
    wait_for_server
    return 0
  fi

  if [[ -n "$STOP_CASE_CMD" ]]; then
    bash -c "$STOP_CASE_CMD"
  fi
  if [[ -n "$RESET_BACKEND_CMD" ]]; then
    bash -c "$RESET_BACKEND_CMD"
  fi
  if [[ -n "$START_CASE_CMD" ]]; then
    bash -c "$START_CASE_CMD"
  elif [[ "$MANUAL_CASE_CONFIRM" == "1" ]]; then
    echo "Start/restart vLLM (and remote_pool daemon if IAXL_RDMA_ENABLE=1) for Case ${CASE_ID} with the variables above, then press Enter."
    read -r _
  fi
  wait_for_server
}

write_summary_header() {
  echo "case_id,case_desc,input_len,concurrency,hit_rate,phase,seed,status,mean_ttft_ms,p50_ttft_ms,p95_ttft_ms,p99_ttft_ms,mean_tpot_ms,req_throughput,output_tok_s,total_tok_s,hits_delta,hits_in_storage_delta,misses_delta,persisted_delta,evicted_delta,auto_spill_rounds_delta,usage_start_pct,usage_end_pct,storage_read_ios_delta,storage_read_mb_delta,storage_read_iops,storage_read_mbps,compress_gbps,decompress_gbps,log_file" > "$SUMMARY"
}

append_summary_row() {
  local case_id=$1
  local case_desc_value=$2
  local input_len=$3
  local concurrency=$4
  local hit_rate=$5
  local phase=$6
  local seed=$7
  local status=$8
  local log=$9
  local start_status=${10}
  local end_status=${11}
  local elapsed=${12}
  local metrics_file=${13}

  python3 - "$SUMMARY" "$case_id" "$case_desc_value" "$input_len" "$concurrency" "$hit_rate" "$phase" "$seed" "$status" "$log" "$start_status" "$end_status" "$elapsed" "$metrics_file" <<'PY'
import csv
import json
import re
import sys

(summary, case_id, case_desc, input_len, concurrency, hit_rate, phase, seed,
 status, log, start_status, end_status, elapsed, metrics_file) = sys.argv[1:]
elapsed = max(float(elapsed), 0.001)

def read_json(path):
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return {}

def workers(data):
    if isinstance(data, dict) and isinstance(data.get("workers"), dict):
        return [v for v in data["workers"].values() if isinstance(v, dict) and "error" not in v]
    return [data] if isinstance(data, dict) else []

def nested(obj, dotted):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur

def number_list(data, field):
    vals = []
    for worker in workers(data):
        value = nested(worker, field)
        if isinstance(value, (int, float, bool)):
            vals.append(float(value))
    return vals

def summed(data, field):
    return sum(number_list(data, field))

def maxed(data, field):
    vals = number_list(data, field)
    return max(vals) if vals else 0.0

def first(data, field):
    vals = number_list(data, field)
    return vals[0] if vals else 0.0

def metric(pattern):
    try:
        text = open(log, "r", encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""
    matches = re.findall(pattern + r"\s*([0-9.]+)", text)
    return matches[-1] if matches else ""

start = read_json(start_status)
end = read_json(end_status)
metrics_data = read_json(metrics_file)

read_ios_delta = max(0.0, first(end, "storage_backend.read_ios_total") - first(start, "storage_backend.read_ios_total"))
read_bytes_delta = max(0.0, first(end, "storage_backend.read_bytes_total") - first(start, "storage_backend.read_bytes_total"))

row = {
    "case_id": case_id,
    "case_desc": case_desc,
    "input_len": input_len,
    "concurrency": concurrency,
    "hit_rate": hit_rate,
    "phase": phase,
    "seed": seed,
    "status": status,
    "mean_ttft_ms": metric(r"Mean TTFT \(ms\):"),
    "p50_ttft_ms": metric(r"P50 TTFT \(ms\):"),
    "p95_ttft_ms": metric(r"P95 TTFT \(ms\):"),
    "p99_ttft_ms": metric(r"P99 TTFT \(ms\):"),
    "mean_tpot_ms": metric(r"Mean TPOT \(ms\):"),
    "req_throughput": metric(r"Request throughput \(req/s\):"),
    "output_tok_s": metric(r"Output token throughput \(tok/s\):"),
    "total_tok_s": metric(r"Total Token throughput \(tok/s\):"),
    "hits_delta": summed(end, "hits") - summed(start, "hits"),
    "hits_in_storage_delta": summed(end, "hits_in_storage") - summed(start, "hits_in_storage"),
    "misses_delta": summed(end, "misses") - summed(start, "misses"),
    "persisted_delta": summed(end, "persisted_total") - summed(start, "persisted_total"),
    "evicted_delta": summed(end, "evicted_total") - summed(start, "evicted_total"),
    "auto_spill_rounds_delta": summed(end, "auto_spill.spill_rounds") - summed(start, "auto_spill.spill_rounds"),
    "usage_start_pct": maxed(start, "usage_pct"),
    "usage_end_pct": maxed(end, "usage_pct"),
    "storage_read_ios_delta": read_ios_delta,
    "storage_read_mb_delta": round(read_bytes_delta / 1024 / 1024, 3),
    "storage_read_iops": round(read_ios_delta / elapsed, 2),
    "storage_read_mbps": round(read_bytes_delta / 1024 / 1024 / elapsed, 2),
    "compress_gbps": round(summed(metrics_data, "compress_gbps"), 3),
    "decompress_gbps": round(summed(metrics_data, "decompress_gbps"), 3),
    "log_file": log,
}
fields = [
    "case_id", "case_desc", "input_len", "concurrency", "hit_rate", "phase", "seed", "status",
    "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms", "p99_ttft_ms", "mean_tpot_ms",
    "req_throughput", "output_tok_s", "total_tok_s",
    "hits_delta", "hits_in_storage_delta", "misses_delta", "persisted_delta", "evicted_delta",
    "auto_spill_rounds_delta", "usage_start_pct", "usage_end_pct",
    "storage_read_ios_delta", "storage_read_mb_delta", "storage_read_iops", "storage_read_mbps",
    "compress_gbps", "decompress_gbps", "log_file",
]
with open(summary, "a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writerow(row)
PY
}

run_one_measure() {
  local case_id=$1
  local desc=$2
  local input_len=$3
  local concurrency=$4
  local hit_rate=$5
  local run_dir=$6
  local seed=$((BASE_SEED + input_len + concurrency + hit_rate))
  local measure_seed=$((seed + MEASURE_SEED_OFFSET))

  mkdir -p "$run_dir"
  echo "--- case=${case_id} input_len=${input_len} concurrency=${concurrency} hit_rate=${hit_rate} ---"

  if [[ "$EVICT_BEFORE_EACH_MEASURE" == "1" && "$case_id" != "A" ]]; then
    evict_cache "$run_dir/evict_before_measure.json"
  fi

  if [[ "$BENCH_WORKLOAD_MODE" == "revisit" ]]; then
    local pressure_seed=$((seed + PRESSURE_SEED_OFFSET))
    echo "  workload=revisit target_seed=${seed} pressure_seed=${pressure_seed} warm_prompts=${REVISIT_WARM_NUM_PROMPTS} pressure_prompts=${PRESSURE_NUM_PROMPTS} revisit_prompts=${REVISIT_MEASURE_NUM_PROMPTS}"

    if [[ "$case_id" != "A" || "$RUN_WARM_FOR_BASELINE" == "1" ]]; then
      if ! run_bench "$run_dir/warm_target.log" "$input_len" "$hit_rate" "$concurrency" "$seed" "$REVISIT_WARM_NUM_PROMPTS"; then
        echo "WARN: target warm phase failed, continuing to pressure/revisit: $run_dir/warm_target.log" >&2
      fi
    fi

    if [[ "$case_id" != "A" || "$RUN_WARM_FOR_BASELINE" == "1" ]]; then
      if ! run_bench "$run_dir/pressure.log" "$input_len" "$hit_rate" "$concurrency" "$pressure_seed" "$PRESSURE_NUM_PROMPTS"; then
        echo "WARN: pressure phase failed, continuing to revisit: $run_dir/pressure.log" >&2
      fi
    fi

    if [[ "$case_id" == "B" || "$case_id" == "C" ]]; then
      sleep "$PRE_CLEANUP_SLEEP"
      wait_flag_false "$run_dir" "auto_spill.running" "$WAIT_SPILL_SECONDS"
      if [[ "$SPILL_BEFORE_REVISIT" == "1" ]]; then
        local evict_only_arg="false"
        [[ "$case_id" == "B" ]] && evict_only_arg="true"
        spill_cache "$run_dir/spill_before_revisit.json" "$evict_only_arg"
        wait_flag_false "$run_dir" "auto_spill.running" "$WAIT_SPILL_SECONDS"
      fi
    fi

    local start_status="$run_dir/status_measure_start.json"
    local end_status="$run_dir/status_measure_end.json"
    local metrics_file="$run_dir/metrics_measure.json"
    local samples="$run_dir/status_samples.jsonl"
    local log="$run_dir/measure.log"

    if [[ "$case_id" != "A" ]]; then
      reset_case_metrics
    fi
    api_status "$start_status"
    monitor_status "$samples" &
    local monitor_pid=$!
    local start_s=$SECONDS
    local status=PASS
    if ! run_bench "$log" "$input_len" "$hit_rate" "$concurrency" "$seed" "$REVISIT_MEASURE_NUM_PROMPTS"; then
      status=FAIL
      echo "WARN: revisit measure failed: $log" >&2
    fi
    local elapsed=$((SECONDS - start_s))
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
    sleep "$PRE_CLEANUP_SLEEP"
    api_status "$end_status"
    collect_case_metrics "$metrics_file"

    append_summary_row "$case_id" "$desc" "$input_len" "$concurrency" "$hit_rate" "measure" "$seed" "$status" "$log" "$start_status" "$end_status" "$elapsed" "$metrics_file"

    local p95 p99 tok
    p95=$(extract_metric "P95 TTFT (ms):" "$log")
    p99=$(extract_metric "P99 TTFT (ms):" "$log")
    tok=$(extract_metric "Output token throughput" "$log")
    echo "  status=${status} p95=${p95:-NA}ms p99=${p99:-NA}ms output_tok_s=${tok:-NA}"

    if [[ "$EVICT_AFTER_EACH_MEASURE" == "1" && "$case_id" != "A" ]]; then
      evict_cache "$run_dir/evict_after_measure.json"
    fi
    return 0
  elif [[ "$BENCH_WORKLOAD_MODE" != "legacy" ]]; then
    echo "ERROR: unsupported BENCH_WORKLOAD_MODE=$BENCH_WORKLOAD_MODE (use revisit/legacy)" >&2
    return 1
  fi

  if [[ "$case_id" != "A" || "$RUN_WARM_FOR_BASELINE" == "1" ]]; then
    if ! run_bench "$run_dir/warm.log" "$input_len" "$hit_rate" "$concurrency" "$seed" "$WARM_NUM_PROMPTS"; then
      echo "WARN: warm phase failed, continuing to measure: $run_dir/warm.log" >&2
    fi
  fi

  if [[ "$case_id" == "B" || "$case_id" == "C" ]]; then
    sleep "$PRE_CLEANUP_SLEEP"
    wait_flag_false "$run_dir" "auto_spill.running" "$WAIT_SPILL_SECONDS"
  fi

  local start_status="$run_dir/status_measure_start.json"
  local end_status="$run_dir/status_measure_end.json"
  local metrics_file="$run_dir/metrics_measure.json"
  local samples="$run_dir/status_samples.jsonl"
  local log="$run_dir/measure.log"

  if [[ "$case_id" != "A" ]]; then
    reset_case_metrics
  fi
  api_status "$start_status"
  monitor_status "$samples" &
  local monitor_pid=$!
  local start_s=$SECONDS
  local status=PASS
  if ! run_bench "$log" "$input_len" "$hit_rate" "$concurrency" "$measure_seed" "$MEASURE_NUM_PROMPTS"; then
    status=FAIL
    echo "WARN: measure failed: $log" >&2
  fi
  local elapsed=$((SECONDS - start_s))
  kill "$monitor_pid" >/dev/null 2>&1 || true
  wait "$monitor_pid" >/dev/null 2>&1 || true
  sleep "$PRE_CLEANUP_SLEEP"
  api_status "$end_status"
  collect_case_metrics "$metrics_file"

  append_summary_row "$case_id" "$desc" "$input_len" "$concurrency" "$hit_rate" "measure" "$measure_seed" "$status" "$log" "$start_status" "$end_status" "$elapsed" "$metrics_file"

  local p95 p99 tok
  p95=$(extract_metric "P95 TTFT (ms):" "$log")
  p99=$(extract_metric "P99 TTFT (ms):" "$log")
  tok=$(extract_metric "Output token throughput" "$log")
  echo "  status=${status} p95=${p95:-NA}ms p99=${p99:-NA}ms output_tok_s=${tok:-NA}"

  if [[ "$EVICT_AFTER_EACH_MEASURE" == "1" && "$case_id" != "A" ]]; then
    evict_cache "$run_dir/evict_after_measure.json"
  fi
}

run_case() {
  local case_id=$1
  local desc
  desc=$(case_desc "$case_id")
  local case_dir="$LOG_DIR/case_${case_id}"
  mkdir -p "$case_dir"
  prepare_case "$case_id"

  if [[ "$EVICT_BETWEEN_CASES" == "1" && "$case_id" != "A" ]]; then
    evict_cache "$case_dir/evict_before_case.json"
  fi

  for hit_rate in $HIT_RATES; do
    for input_len in $INPUT_LENS; do
      local concurrencies
      concurrencies=$(concurrencies_for_len "$input_len")
      for concurrency in $concurrencies; do
        local run_dir="$case_dir/in${input_len}_con${concurrency}_hit${hit_rate}"
        run_one_measure "$case_id" "$desc" "$input_len" "$concurrency" "$hit_rate" "$run_dir"
      done
    done
  done
}

generate_report() {
  python3 - "$SUMMARY" "$COMPARE" "$REPORT" "$P95_IMPROVEMENT_TARGET_PCT" "$P99_IMPROVEMENT_TARGET_PCT" "$THROUGHPUT_DROP_TOLERANCE_PCT" "$REQUIRE_STORAGE_HITS" <<'PY'
import csv
import math
import sys
from collections import defaultdict

summary, compare, report = sys.argv[1:4]
p95_target = float(sys.argv[4])
p99_target = float(sys.argv[5])
tp_tol = float(sys.argv[6])
need_storage = sys.argv[7].lower() in ("1", "true", "yes", "on")

def num(row, key):
    try:
        value = row.get(key, "")
        if value in ("", "FAIL"):
            return math.nan
        return float(value)
    except Exception:
        return math.nan

rows = []
with open(summary, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if row.get("phase") == "measure" and row.get("status") == "PASS":
            rows.append(row)

by_key = defaultdict(dict)
for row in rows:
    by_key[(row["input_len"], row["concurrency"], row["hit_rate"])][row["case_id"]] = row

fields = [
    "input_len", "concurrency", "hit_rate",
    "A_p95_ms", "B_p95_ms", "C_p95_ms", "A_to_C_p95_improve_pct", "B_to_C_p95_improve_pct",
    "A_p99_ms", "B_p99_ms", "C_p99_ms", "A_to_C_p99_improve_pct", "B_to_C_p99_improve_pct",
    "A_output_tok_s", "B_output_tok_s", "C_output_tok_s", "A_to_C_output_tok_s_change_pct",
    "C_hits_in_storage_delta", "C_storage_read_mbps", "verdict",
]
compare_rows = []
for key, cases in sorted(by_key.items(), key=lambda item: tuple(int(x) for x in item[0])):
    if "C" not in cases or ("A" not in cases and "B" not in cases):
        continue
    c = cases["C"]
    a = cases.get("A", {})
    b = cases.get("B", {})
    a_p95, b_p95, c_p95 = num(a, "p95_ttft_ms"), num(b, "p95_ttft_ms"), num(c, "p95_ttft_ms")
    a_p99, b_p99, c_p99 = num(a, "p99_ttft_ms"), num(b, "p99_ttft_ms"), num(c, "p99_ttft_ms")
    a_tp, b_tp, c_tp = num(a, "output_tok_s"), num(b, "output_tok_s"), num(c, "output_tok_s")

    a_to_c_p95 = (a_p95 - c_p95) / a_p95 * 100 if a_p95 > 0 else math.nan
    b_to_c_p95 = (b_p95 - c_p95) / b_p95 * 100 if b_p95 > 0 else math.nan
    a_to_c_p99 = (a_p99 - c_p99) / a_p99 * 100 if a_p99 > 0 else math.nan
    b_to_c_p99 = (b_p99 - c_p99) / b_p99 * 100 if b_p99 > 0 else math.nan
    tp_change = (c_tp - a_tp) / a_tp * 100 if a_tp > 0 else ((c_tp - b_tp) / b_tp * 100 if b_tp > 0 else math.nan)
    storage_hits = num(c, "hits_in_storage_delta")

    ref_p95_imp = a_to_c_p95 if not math.isnan(a_to_c_p95) else b_to_c_p95
    ref_p99_imp = a_to_c_p99 if not math.isnan(a_to_c_p99) else b_to_c_p99
    ok = (
        ref_p95_imp >= p95_target
        and ref_p99_imp >= p99_target
        and tp_change >= -tp_tol
        and (not need_storage or storage_hits > 0)
    )
    compare_rows.append({
        "input_len": key[0], "concurrency": key[1], "hit_rate": key[2],
        "A_p95_ms": a_p95, "B_p95_ms": b_p95, "C_p95_ms": c_p95,
        "A_to_C_p95_improve_pct": a_to_c_p95, "B_to_C_p95_improve_pct": b_to_c_p95,
        "A_p99_ms": a_p99, "B_p99_ms": b_p99, "C_p99_ms": c_p99,
        "A_to_C_p99_improve_pct": a_to_c_p99, "B_to_C_p99_improve_pct": b_to_c_p99,
        "A_output_tok_s": a_tp, "B_output_tok_s": b_tp, "C_output_tok_s": c_tp,
        "A_to_C_output_tok_s_change_pct": tp_change,
        "C_hits_in_storage_delta": storage_hits,
        "C_storage_read_mbps": num(c, "storage_read_mbps"),
        "verdict": "PASS" if ok else "FAIL",
    })

with open(compare, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(compare_rows)

passed = sum(1 for row in compare_rows if row["verdict"] == "PASS")
with open(report, "w", encoding="utf-8") as f:
    f.write("# KVShrink Tiered Storage PoC Conclusion (Case A / B / C)\n\n")
    f.write(f"Comparable points: {len(compare_rows)}\n")
    f.write(f"Passed points: {passed}/{len(compare_rows)}\n\n")
    f.write("Criteria:\n")
    f.write(f"- Case C P95 TTFT improvement >= {p95_target:.1f}%\n")
    f.write(f"- Case C P99 TTFT improvement >= {p99_target:.1f}%\n")
    f.write(f"- Case C output throughput drop <= {tp_tol:.1f}%\n")
    f.write(f"- Case C hits_in_storage_delta > 0 required: {need_storage}\n\n")
    if not compare_rows:
        f.write("Conclusion:SINGLE-CASE / INCONCLUSIVE. Run Case A/B and Case C to generate cross-case comparison rows.\n")
    elif passed == len(compare_rows):
        f.write("Conclusion: PASS. Case C satisfies the configured PoC criteria for every comparable point.\n")
    elif passed > 0:
        f.write("Conclusion: PARTIAL PASS. Some matrix points satisfy the criteria; inspect case_comparison.csv for details.\n")
    else:
        f.write("Conclusion: FAIL. Case C did not satisfy the configured criteria.\n")
PY
}

write_metadata() {
  {
    echo "MODEL=$MODEL"
    echo "HOST=$HOST"
    echo "PORT=$PORT"
    echo "REMOTE_CACHE_HOST=$REMOTE_CACHE_HOST"
    echo "REMOTE_CACHE_PORT=$REMOTE_CACHE_PORT"
    echo "API_BASE=$API_BASE"
    echo "SERVER_LIFECYCLE_MODE=$SERVER_LIFECYCLE_MODE"
    echo "LIVE_SERVER_CASE_HINT=$LIVE_SERVER_CASE_HINT"
    echo "BENCH_RUNNER=$BENCH_RUNNER"
    echo "BENCH_CONTAINER_NAME=$BENCH_CONTAINER_NAME"
    echo "CASES=$CASES"
    echo "BENCH_WORKLOAD_MODE=$BENCH_WORKLOAD_MODE"
    echo "INPUT_LENS=$INPUT_LENS"
    echo "HIT_RATES=$HIT_RATES"
    echo "OUTPUT_LEN=$OUTPUT_LEN"
    echo "REVISIT_WARM_NUM_PROMPTS=$REVISIT_WARM_NUM_PROMPTS"
    echo "PRESSURE_NUM_PROMPTS=$PRESSURE_NUM_PROMPTS"
    echo "REVISIT_MEASURE_NUM_PROMPTS=$REVISIT_MEASURE_NUM_PROMPTS"
    echo "IAXL_DDR_POOL_SIZE_GB=$IAXL_DDR_POOL_SIZE_GB"
    echo "IAXL_CACHE_DIR=$IAXL_CACHE_DIR"
    echo "IAXL_KV_COMPRESSION=$IAXL_KV_COMPRESSION"
    echo "IAXL_CACHE_AUTO_SPILL_ENABLE=$IAXL_CACHE_AUTO_SPILL_ENABLE"
    echo "IAXL_CACHE_AUTO_SPILL_EVICT_ONLY=$IAXL_CACHE_AUTO_SPILL_EVICT_ONLY"
    echo "IAXL_CACHE_HIGH_WATERMARK=$IAXL_CACHE_HIGH_WATERMARK"
    echo "IAXL_CACHE_LOW_WATERMARK=$IAXL_CACHE_LOW_WATERMARK"
    echo "IAXL_CACHE_SPILL_BATCH_GROUPS=$IAXL_CACHE_SPILL_BATCH_GROUPS"
    echo "IAXL_CACHE_SPILL_PACE_MS=$IAXL_CACHE_SPILL_PACE_MS"
    echo "IAXL_CACHE_EVICT_GRACE_MS=$IAXL_CACHE_EVICT_GRACE_MS"
  } > "$META"
}

resolve_server_lifecycle_mode
resolve_bench_runner
write_summary_header
write_metadata

PASS=0
FAIL=0
for case_id in $CASES; do
  if run_case "$case_id"; then
    PASS=$((PASS + 1))
  else
    FAIL=$((FAIL + 1))
    echo "ERROR: case $case_id failed; continuing" >&2
  fi
done

if [[ -n "$STOP_CASE_CMD" ]]; then
  bash -c "$STOP_CASE_CMD" || true
fi

generate_report

echo "================================================================"
echo "Tiering PoC benchmark done: ${PASS} case groups completed, ${FAIL} failed."
echo "Logs:       $LOG_DIR/"
echo "Summary:    $SUMMARY"
echo "Comparison: $COMPARE"
echo "Conclusion: $REPORT"
echo "================================================================"
cat "$REPORT"
