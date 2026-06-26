#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/baselines/configs/cpid_target_position_acceleration_eval_delay_free.yaml}"
EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-500}"
DELAYS="${DELAYS:-0 2 4 6 8 10 12}"
RUN_NOMINAL="${RUN_NOMINAL:-true}"
RUN_DISTURBANCE="${RUN_DISTURBANCE:-true}"
LOG_ROOT_NOMINAL="${LOG_ROOT_NOMINAL:-${PROJECT_ROOT}/logs/cpid_acceleration_delay_impact_wo_comp}"
LOG_ROOT_DISTURBANCE="${LOG_ROOT_DISTURBANCE:-${PROJECT_ROOT}/logs/cpid_acceleration_delay_impact_wo_comp_disturbance}"
TERMINAL_OUTPUT_DIR_NAME="${TERMINAL_OUTPUT_DIR_NAME:-terminal_outputs}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-5}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-10m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-10s}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-3s}"
NO_SAVE_ROLLOUT="${NO_SAVE_ROLLOUT:-true}"

if [[ -n "${CONDA_ENV}" ]]; then
    PYTHON_CMD=(conda run -n "${CONDA_ENV}" python)
else
    PYTHON_CMD=("${PYTHON_BIN}")
fi

read -r -a DELAY_STEPS <<< "${DELAYS}"

cleanup_process_group() {
    local pgid="$1"
    if [[ "${CLEANUP_AFTER_RUN}" != "true" ]]; then
        return
    fi
    if [[ -z "${pgid}" ]]; then
        return
    fi
    if kill -0 "-${pgid}" 2>/dev/null; then
        echo "[INFO] Cleaning remaining processes in process group ${pgid}"
        kill -TERM "-${pgid}" 2>/dev/null || true
        sleep "${CLEANUP_GRACE_PERIOD}"
        kill -KILL "-${pgid}" 2>/dev/null || true
    fi
}

run_scenario() {
    local scenario_name="$1"
    local log_root="$2"
    local disturbance_enabled="$3"
    local run_suffix="$4"
    local terminal_output_dir="${log_root}/${TERMINAL_OUTPUT_DIR_NAME}"
    local status_file="${log_root}/eval_status.tsv"
    local scenario_status=0
    shift 4

    mkdir -p "${terminal_output_dir}"
    printf "scenario\tdelay_step\trun_name\tstatus\tlog_file\n" > "${status_file}"

    echo "[INFO] Scenario: ${scenario_name}"
    echo "[INFO] Results root: ${log_root}"
    echo "[INFO] Terminal logs: ${terminal_output_dir}"
    echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

    for delay_step in "${DELAY_STEPS[@]}"; do
        local run_name="cpid_target_position_acceleration_delay_${delay_step}${run_suffix}"
        local log_file="${terminal_output_dir}/${run_name}.log"
        local cmd_args=(
            "${PROJECT_ROOT}/scripts/cpid_acceleration_policy_eval.py"
            --config "${CONFIG}"
            --headless
            --episodes "${EPISODES}"
            --num_envs "${NUM_ENVS}"
            --delay_step "${delay_step}"
            --log_root "${log_root}"
            --run_name "${run_name}"
        )

        if [[ "${disturbance_enabled}" == "true" ]]; then
            cmd_args+=(--external_disturbance_enabled)
        fi
        if [[ "${NO_SAVE_ROLLOUT}" == "true" ]]; then
            cmd_args+=(--no_save_rollout)
        fi

        echo "[INFO] Evaluating ${scenario_name}: delay_step=${delay_step}"
        echo "[INFO] Run name: ${run_name}"
        echo "[INFO] Log file: ${log_file}"

        PYTHONUNBUFFERED=1 setsid timeout --kill-after="${TIMEOUT_KILL_AFTER}" "${EVAL_TIMEOUT}" \
            "${PYTHON_CMD[@]}" "${cmd_args[@]}" "$@" > "${log_file}" 2>&1 &
        eval_pid=$!
        wait "${eval_pid}"
        status=$?
        cleanup_process_group "${eval_pid}"

        if [[ ${status} -eq 0 ]]; then
            echo "[INFO] Finished ${run_name}"
            status_text="ok"
        elif [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
            echo "[ERROR] Evaluation timed out or was killed for ${run_name} after ${EVAL_TIMEOUT}" >&2
            status_text="timeout_or_killed:${status}"
            scenario_status=${status}
        else
            echo "[ERROR] Evaluation failed for ${run_name} with exit code ${status}" >&2
            status_text="failed:${status}"
            scenario_status=${status}
        fi

        printf "%s\t%s\t%s\t%s\t%s\n" \
            "${scenario_name}" "${delay_step}" "${run_name}" "${status_text}" "${log_file}" >> "${status_file}"

        echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
        sleep "${SLEEP_BETWEEN_RUNS}"
    done

    echo "[INFO] Scenario complete: ${scenario_name}. Status summary: ${status_file}"
    return "${scenario_status}"
}

if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config file does not exist: ${CONFIG}" >&2
    exit 1
fi

if [[ ${#DELAY_STEPS[@]} -eq 0 ]]; then
    echo "[ERROR] DELAYS is empty." >&2
    exit 1
fi

if [[ "${RUN_NOMINAL}" != "true" && "${RUN_DISTURBANCE}" != "true" ]]; then
    echo "[ERROR] Both RUN_NOMINAL and RUN_DISTURBANCE are disabled." >&2
    exit 1
fi

overall_status=0

echo "[INFO] Acceleration CPID delay-impact sweep"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python command: ${PYTHON_CMD[*]}"
echo "[INFO] Config: ${CONFIG}"
echo "[INFO] Episodes: ${EPISODES}; num_envs: ${NUM_ENVS}; delays: ${DELAYS}"
echo "[INFO] NO_SAVE_ROLLOUT=${NO_SAVE_ROLLOUT}"

if [[ "${RUN_NOMINAL}" == "true" ]]; then
    run_scenario "nominal" "${LOG_ROOT_NOMINAL}" "false" "" "$@"
    status=$?
    if [[ ${status} -ne 0 ]]; then
        overall_status=${status}
    fi
fi

if [[ "${RUN_DISTURBANCE}" == "true" ]]; then
    run_scenario "disturbance" "${LOG_ROOT_DISTURBANCE}" "true" "_disturbance" "$@"
    status=$?
    if [[ ${status} -ne 0 ]]; then
        overall_status=${status}
    fi
fi

echo "[INFO] Acceleration CPID delay-impact sweep complete."
exit "${overall_status}"
