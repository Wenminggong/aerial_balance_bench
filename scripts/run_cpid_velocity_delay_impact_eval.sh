#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CONDA_ENV="${CONDA_ENV:-}"
POLICY_CONFIG="${POLICY_CONFIG:-${PROJECT_ROOT}/baselines/configs/cpid_velocity_max_acc_5_best.yaml}"
EPISODES="${EPISODES:-500}"
NUM_ENVS="${NUM_ENVS:-500}"
DELAYS="${DELAYS:-0 2 4 6 8 10 12}"
RUN_NOMINAL="${RUN_NOMINAL:-true}"
RUN_DISTURBANCE="${RUN_DISTURBANCE:-true}"
LOG_ROOT_NOMINAL="${LOG_ROOT_NOMINAL:-${PROJECT_ROOT}/logs/cpid_velocity_delay_impact_wo_comp}"
LOG_ROOT_DISTURBANCE="${LOG_ROOT_DISTURBANCE:-${PROJECT_ROOT}/logs/cpid_velocity_delay_impact_wo_comp_disturbance}"
TERMINAL_OUTPUT_DIR_NAME="${TERMINAL_OUTPUT_DIR_NAME:-terminal_outputs}"
CONFIG_DIR_NAME="${CONFIG_DIR_NAME:-generated_configs}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-5}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-10m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-10s}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-3s}"
NO_SAVE_ROLLOUT="${NO_SAVE_ROLLOUT:-true}"
MAX_ACC="${MAX_ACC:-5.0}"
MAX_VELOCITY="${MAX_VELOCITY:-0.0}"
EXTERNAL_DISTURBANCE_OU_CLIP="${EXTERNAL_DISTURBANCE_OU_CLIP:-0.1}"

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

write_env_config() {
    local path="$1"
    local delay_step="$2"
    local disturbance_enabled="$3"
    local log_root="$4"
    local run_name="$5"
    local robustness_enabled="false"
    local action_delay_enabled="false"

    if [[ "${delay_step}" -gt 0 ]]; then
        robustness_enabled="true"
        action_delay_enabled="true"
    fi
    if [[ "${disturbance_enabled}" == "true" ]]; then
        robustness_enabled="true"
    fi

    cat > "${path}" <<EOF
task_name: target_position
interface_name: velocity

env:
  seed: 666
  num_envs: ${NUM_ENVS}
  episode_length_s: 10.0
  device: cuda:0

target_position_task:
  random_goal: true
  fixed_goal_position: 0.35
  ball_position_min: 0.10
  ball_position_max: 0.60
  goal_position_min: 0.10
  goal_position_max: 0.60
  min_initial_goal_distance: 0.05

velocity_interface:
  max_acc: ${MAX_ACC}
  max_velocity: ${MAX_VELOCITY}

robustness:
  enabled: ${robustness_enabled}
  ball_mass_variation_enabled: false
  ball_mass_range: [0.005, 0.05]
  recompute_ball_inertia: true
  controller_gain_variation_enabled: false
  controller_gain_range: [4.0, 8.0]
  action_delay_enabled: ${action_delay_enabled}
  delay_step: ${delay_step}
  external_disturbance_enabled: ${disturbance_enabled}
  external_disturbance_ou_mu: 0.0
  external_disturbance_ou_theta_range: [0.1, 0.3]
  external_disturbance_ou_sigma_range: [0.01, 0.05]
  external_disturbance_ou_clip: ${EXTERNAL_DISTURBANCE_OU_CLIP}

target_position_evaluator:
  episode_length_s: 10.0
  num_eval_episodes: ${EPISODES}
  target_zone: 0.01
  steady_state_window_s: 1.0

runner:
  target_episodes: ${EPISODES}
  max_steps: null
  render: false
  save_rollout: false

logging:
  root_dir: ${log_root}
  run_name: ${run_name}
EOF
}

write_run_config() {
    local path="$1"
    local env_config="$2"
    local log_root="$3"
    local run_name="$4"
    local save_rollout="true"

    if [[ "${NO_SAVE_ROLLOUT}" == "true" ]]; then
        save_rollout="false"
    fi

    cat > "${path}" <<EOF
env_config: ${env_config}
policy_config: ${POLICY_CONFIG}

runner:
  target_episodes: ${EPISODES}
  max_steps: null
  render: false
  save_rollout: ${save_rollout}

logging:
  root_dir: ${log_root}
  run_name: ${run_name}
EOF
}

run_scenario() {
    local scenario_name="$1"
    local log_root="$2"
    local disturbance_enabled="$3"
    local run_suffix="$4"
    local terminal_output_dir="${log_root}/${TERMINAL_OUTPUT_DIR_NAME}"
    local config_dir="${log_root}/${CONFIG_DIR_NAME}"
    local status_file="${log_root}/eval_status.tsv"
    local scenario_status=0
    shift 4

    mkdir -p "${terminal_output_dir}" "${config_dir}"
    printf "scenario\tdelay_step\trun_name\tstatus\tlog_file\trun_config\n" > "${status_file}"

    echo "[INFO] Scenario: ${scenario_name}"
    echo "[INFO] Results root: ${log_root}"
    echo "[INFO] Generated configs: ${config_dir}"
    echo "[INFO] Terminal logs: ${terminal_output_dir}"
    echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

    for delay_step in "${DELAY_STEPS[@]}"; do
        local run_name="cpid_target_position_velocity_delay_${delay_step}${run_suffix}"
        local env_config="${config_dir}/${run_name}_env.yaml"
        local run_config="${config_dir}/${run_name}_run.yaml"
        local log_file="${terminal_output_dir}/${run_name}.log"

        write_env_config "${env_config}" "${delay_step}" "${disturbance_enabled}" "${log_root}" "${run_name}"
        write_run_config "${run_config}" "${env_config}" "${log_root}" "${run_name}"

        local cmd_args=(
            "${PROJECT_ROOT}/scripts/cpid_policy_eval.py"
            --config "${run_config}"
            --headless
            --episodes "${EPISODES}"
            --num_envs "${NUM_ENVS}"
        )

        echo "[INFO] Evaluating ${scenario_name}: delay_step=${delay_step}"
        echo "[INFO] Run name: ${run_name}"
        echo "[INFO] Run config: ${run_config}"
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

        printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
            "${scenario_name}" "${delay_step}" "${run_name}" "${status_text}" "${log_file}" "${run_config}" >> "${status_file}"

        echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
        sleep "${SLEEP_BETWEEN_RUNS}"
    done

    echo "[INFO] Scenario complete: ${scenario_name}. Status summary: ${status_file}"
    return "${scenario_status}"
}

if [[ ! -f "${POLICY_CONFIG}" ]]; then
    echo "[ERROR] Policy config file does not exist: ${POLICY_CONFIG}" >&2
    exit 1
fi

if [[ ${#DELAY_STEPS[@]} -eq 0 ]]; then
    echo "[ERROR] DELAYS is empty." >&2
    exit 1
fi

for delay_step in "${DELAY_STEPS[@]}"; do
    if ! [[ "${delay_step}" =~ ^[0-9]+$ ]]; then
        echo "[ERROR] Delay must be a non-negative integer, got: ${delay_step}" >&2
        exit 1
    fi
done

if [[ "${RUN_NOMINAL}" != "true" && "${RUN_DISTURBANCE}" != "true" ]]; then
    echo "[ERROR] Both RUN_NOMINAL and RUN_DISTURBANCE are disabled." >&2
    exit 1
fi

overall_status=0

echo "[INFO] Velocity CPID delay-impact sweep without state-predictor compensation"
echo "[INFO] Project root: ${PROJECT_ROOT}"
echo "[INFO] Python command: ${PYTHON_CMD[*]}"
echo "[INFO] Policy config: ${POLICY_CONFIG}"
echo "[INFO] Episodes: ${EPISODES}; num_envs: ${NUM_ENVS}; delays: ${DELAYS}"
echo "[INFO] MAX_ACC=${MAX_ACC}; NO_SAVE_ROLLOUT=${NO_SAVE_ROLLOUT}"

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

echo "[INFO] Velocity CPID delay-impact sweep complete."
exit "${overall_status}"
