#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_ROOT}/baselines/configs/nffb_unified_tracking_eval.yaml}"
TARGET_EPISODES="${TARGET_EPISODES:-1000}"
NUM_ENVS="${NUM_ENVS:-10}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/logs/nffb/unified_tracking/eval_logs}"
SUMMARY_FILE="${RESULTS_ROOT}/eval_status.tsv"

if [[ -n "${REFERENCE_TYPES:-}" ]]; then
    read -r -a TYPES <<< "${REFERENCE_TYPES}"
else
    TYPES=(constant sine triangle trapezoid random_b_spline random_ramp_dwell)
fi

mkdir -p "${RESULTS_ROOT}"
printf "reference_type\tstatus\tlog_file\n" > "${SUMMARY_FILE}"
overall_status=0

for reference_type in "${TYPES[@]}"; do
    env_config="${PROJECT_ROOT}/environments/configs/unified_tracking_${reference_type}.yaml"
    log_file="${RESULTS_ROOT}/nffb_unified_tracking_${reference_type}.log"
    if [[ ! -f "${env_config}" ]]; then
        echo "[ERROR] Missing environment config: ${env_config}" >&2
        exit 1
    fi

    echo "[INFO] Evaluating NFFB on ${reference_type}"
    "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/nffb_policy_eval.py" \
        --config "${RUN_CONFIG}" \
        --env_config "${env_config}" \
        --episodes "${TARGET_EPISODES}" \
        --num_envs "${NUM_ENVS}" \
        --run_name "nffb_unified_tracking_${reference_type}" \
        "$@" > "${log_file}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
        status_text="ok"
    else
        status_text="failed:${status}"
        overall_status=${status}
        echo "[ERROR] ${reference_type} evaluation failed with status ${status}" >&2
    fi
    printf "%s\t%s\t%s\n" "${reference_type}" "${status_text}" "${log_file}" >> "${SUMMARY_FILE}"
done

echo "[INFO] NFFB evaluation sweep complete: ${SUMMARY_FILE}"
exit "${overall_status}"
