#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/logs/rl/rpo_models}"
MODEL_NAME="${MODEL_NAME:-}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/baselines/configs/rl_target_position_rpo_eval_delay_free.yaml}"
RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/logs/rl/rpo_results}"
SUMMARY_FILE="${RESULTS_ROOT}/eval_status.tsv"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-5}"
EVAL_TIMEOUT="${EVAL_TIMEOUT:-10m}"
TIMEOUT_KILL_AFTER="${TIMEOUT_KILL_AFTER:-5s}"
CLEANUP_AFTER_RUN="${CLEANUP_AFTER_RUN:-true}"
CLEANUP_GRACE_PERIOD="${CLEANUP_GRACE_PERIOD:-2s}"
TARGET_EPISODES_VALUE="${TARGET_EPISODES:-config_default}"
NUM_ENVS_VALUE="${NUM_ENVS:-config_default}"

# ENV_CONFIGS and RUN_NAMES are paired by array index. Every run name is passed
# through unchanged; the script never appends model, config, or environment names.
ENV_CONFIGS=(
    # "${PROJECT_ROOT}/environments/configs/template_eval_delay_free.yaml"
    # "${PROJECT_ROOT}/environments/configs/template_eval_delay_15.yaml"
)
RUN_NAMES=(
    # "delay_free_eval"
    # "delay_15_eval"
)

cleanup_process_group() {
    local pgid="$1"
    if [[ "${CLEANUP_AFTER_RUN}" != "true" || -z "${pgid}" ]]; then
        return
    fi
    if kill -0 "-${pgid}" 2>/dev/null; then
        echo "[INFO] Cleaning remaining processes in process group ${pgid}"
        kill -TERM "-${pgid}" 2>/dev/null || true
        sleep "${CLEANUP_GRACE_PERIOD}"
        kill -KILL "-${pgid}" 2>/dev/null || true
    fi
}

CLI_ENV_CONFIGS=()
CLI_RUN_NAMES=()
FORWARD_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env_config)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "[ERROR] --env_config requires a path." >&2
                exit 1
            fi
            CLI_ENV_CONFIGS+=("$2")
            shift 2
            ;;
        --env_config=*)
            value="${1#*=}"
            if [[ -z "${value}" ]]; then
                echo "[ERROR] --env_config requires a path." >&2
                exit 1
            fi
            CLI_ENV_CONFIGS+=("${value}")
            shift
            ;;
        --run_name)
            if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
                echo "[ERROR] --run_name requires a name." >&2
                exit 1
            fi
            CLI_RUN_NAMES+=("$2")
            shift 2
            ;;
        --run_name=*)
            value="${1#*=}"
            if [[ -z "${value}" ]]; then
                echo "[ERROR] --run_name requires a name." >&2
                exit 1
            fi
            CLI_RUN_NAMES+=("${value}")
            shift
            ;;
        --episodes|--episodes=*)
            echo "[ERROR] Do not pass --episodes directly. Use TARGET_EPISODES=<positive-int>." >&2
            exit 1
            ;;
        --num_envs|--num_envs=*)
            echo "[ERROR] Do not pass --num_envs directly. Use NUM_ENVS=<positive-int>." >&2
            exit 1
            ;;
        *)
            FORWARD_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#CLI_ENV_CONFIGS[@]} -gt 0 || ${#CLI_RUN_NAMES[@]} -gt 0 ]]; then
    ENV_CONFIGS=("${CLI_ENV_CONFIGS[@]}")
    RUN_NAMES=("${CLI_RUN_NAMES[@]}")
elif [[ -n "${ENV_CONFIG:-}" || -n "${RUN_NAME:-}" ]]; then
    if [[ -z "${ENV_CONFIG:-}" || -z "${RUN_NAME:-}" ]]; then
        echo "[ERROR] ENV_CONFIG and RUN_NAME must be provided together." >&2
        exit 1
    fi
    ENV_CONFIGS=("${ENV_CONFIG}")
    RUN_NAMES=("${RUN_NAME}")
fi

if [[ ${#ENV_CONFIGS[@]} -eq 0 ]]; then
    echo "[ERROR] At least one environment config is required." >&2
    echo "[INFO] Configure ENV_CONFIGS/RUN_NAMES in the script or pass paired --env_config/--run_name options." >&2
    exit 1
fi
if [[ ${#ENV_CONFIGS[@]} -ne ${#RUN_NAMES[@]} ]]; then
    echo "[ERROR] ENV_CONFIGS and RUN_NAMES must have the same number of entries." >&2
    echo "[INFO] ENV_CONFIGS=${#ENV_CONFIGS[@]}; RUN_NAMES=${#RUN_NAMES[@]}" >&2
    exit 1
fi

EVAL_OVERRIDES=()
if [[ -n "${TARGET_EPISODES:-}" ]]; then
    if ! [[ "${TARGET_EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] TARGET_EPISODES must be a positive integer, got '${TARGET_EPISODES}'." >&2
        exit 1
    fi
    EVAL_OVERRIDES+=(--episodes "${TARGET_EPISODES}")
fi
if [[ -n "${NUM_ENVS:-}" ]]; then
    if ! [[ "${NUM_ENVS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "[ERROR] NUM_ENVS must be a positive integer, got '${NUM_ENVS}'." >&2
        exit 1
    fi
    EVAL_OVERRIDES+=(--num_envs "${NUM_ENVS}")
fi

if [[ "${MODEL_ROOT}" != /* ]]; then
    MODEL_ROOT="${PROJECT_ROOT}/${MODEL_ROOT}"
fi
if [[ ! -d "${MODEL_ROOT}" ]]; then
    echo "[ERROR] Model root directory does not exist: ${MODEL_ROOT}" >&2
    exit 1
fi

if [[ -f "${MODEL_ROOT}/checkpoints/best_agent.pt" ]]; then
    MODEL_DIR="${MODEL_ROOT}"
    CHECKPOINT="${MODEL_ROOT}/checkpoints/best_agent.pt"
    MODEL_ROOT_MODE="single_model"
elif [[ -n "${MODEL_NAME}" ]]; then
    MODEL_DIR="${MODEL_ROOT}/${MODEL_NAME}"
    CHECKPOINT="${MODEL_DIR}/checkpoints/best_agent.pt"
    MODEL_ROOT_MODE="model_collection_selection"
    if [[ ! -f "${CHECKPOINT}" ]]; then
        echo "[ERROR] Selected model checkpoint does not exist: ${CHECKPOINT}" >&2
        exit 1
    fi
else
    mapfile -t CHECKPOINTS < <(
        find "${MODEL_ROOT}" -mindepth 3 -maxdepth 3 -type f -path "*/checkpoints/best_agent.pt" | sort
    )
    if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
        echo "[ERROR] No best_agent.pt checkpoint found under: ${MODEL_ROOT}" >&2
        exit 1
    fi
    if [[ ${#CHECKPOINTS[@]} -gt 1 ]]; then
        echo "[ERROR] MODEL_ROOT contains ${#CHECKPOINTS[@]} models, but this script evaluates exactly one model." >&2
        echo "[INFO] Set MODEL_NAME=<model-directory-name> or point MODEL_ROOT directly to one model directory." >&2
        exit 1
    fi
    CHECKPOINT="${CHECKPOINTS[0]}"
    MODEL_DIR="$(dirname "$(dirname "${CHECKPOINT}")")"
    MODEL_ROOT_MODE="single_model_collection"
fi
MODEL_NAME_RESOLVED="$(basename "${MODEL_DIR}")"

if [[ "${CONFIG}" != /* ]]; then
    CONFIG="${PROJECT_ROOT}/${CONFIG}"
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "[ERROR] Config file does not exist: ${CONFIG}" >&2
    exit 1
fi
CONFIG_NAME="$(basename "${CONFIG}" .yaml)"

declare -A SEEN_RUN_NAMES=()
for idx in "${!ENV_CONFIGS[@]}"; do
    if [[ "${ENV_CONFIGS[$idx]}" != /* ]]; then
        ENV_CONFIGS[$idx]="${PROJECT_ROOT}/${ENV_CONFIGS[$idx]}"
    fi
    if [[ ! -f "${ENV_CONFIGS[$idx]}" ]]; then
        echo "[ERROR] Environment config file does not exist: ${ENV_CONFIGS[$idx]}" >&2
        exit 1
    fi

    run_name="${RUN_NAMES[$idx]}"
    if [[ -z "${run_name}" ]]; then
        echo "[ERROR] RUN_NAMES[$idx] must not be empty." >&2
        exit 1
    fi
    if [[ "${run_name}" == */* || "${run_name}" == "." || "${run_name}" == ".." || \
          "${run_name}" == *$'\t'* || "${run_name}" == *$'\n'* ]]; then
        echo "[ERROR] Invalid run name '${run_name}': use a single directory name without '/', tabs, or newlines." >&2
        exit 1
    fi
    if [[ -n "${SEEN_RUN_NAMES[${run_name}]+x}" ]]; then
        echo "[ERROR] Duplicate run name '${run_name}'. Every run name must be unique to prevent output overwrite." >&2
        exit 1
    fi
    SEEN_RUN_NAMES["${run_name}"]=1
done

mkdir -p "${RESULTS_ROOT}"
CONFIG_RESULTS_ROOT="${RESULTS_ROOT}/${CONFIG_NAME}"
CONFIG_SUMMARY_FILE="${CONFIG_RESULTS_ROOT}/eval_status.tsv"
mkdir -p "${CONFIG_RESULTS_ROOT}"

SUMMARY_HEADER="config_name\tconfig_path\tenv_config_name\tenv_config_path\tmodel_name\tmodel_dir\trun_name\tcheckpoint\ttarget_episodes\tnum_envs\tstatus\tlog_file"
printf "%b\n" "${SUMMARY_HEADER}" > "${SUMMARY_FILE}"
printf "%b\n" "${SUMMARY_HEADER}" > "${CONFIG_SUMMARY_FILE}"

overall_status=0
echo "[INFO] Evaluating exactly one model: ${MODEL_NAME_RESOLVED}"
echo "[INFO] MODEL_ROOT=${MODEL_ROOT} (${MODEL_ROOT_MODE})"
echo "[INFO] Checkpoint: ${CHECKPOINT}"
echo "[INFO] Config: ${CONFIG_NAME} (${CONFIG})"
echo "[INFO] Found ${#ENV_CONFIGS[@]} environment/run-name pair(s)."
echo "[INFO] Results will be written to: ${RESULTS_ROOT}"
echo "[INFO] TARGET_EPISODES=${TARGET_EPISODES_VALUE}; NUM_ENVS=${NUM_ENVS_VALUE}"

for idx in "${!ENV_CONFIGS[@]}"; do
    env_config="${ENV_CONFIGS[$idx]}"
    env_config_name="$(basename "${env_config}" .yaml)"
    run_name="${RUN_NAMES[$idx]}"
    env_results_root="${CONFIG_RESULTS_ROOT}/${env_config_name}"
    log_file="${env_results_root}/${run_name}.log"
    mkdir -p "${env_results_root}"

    echo "[INFO] Evaluating run: ${run_name}"
    echo "[INFO] Environment config: ${env_config_name} (${env_config})"
    echo "[INFO] Log file: ${log_file}"
    echo "[INFO] Eval timeout: ${EVAL_TIMEOUT} (kill after ${TIMEOUT_KILL_AFTER})"

    PYTHONUNBUFFERED=1 setsid timeout --kill-after="${TIMEOUT_KILL_AFTER}" "${EVAL_TIMEOUT}" \
        "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/rl_policy_eval.py" \
        --config "${CONFIG}" \
        --env_config "${env_config}" \
        --checkpoint "${CHECKPOINT}" \
        --run_name "${run_name}" \
        "${EVAL_OVERRIDES[@]}" \
        "${FORWARD_ARGS[@]}" > "${log_file}" 2>&1 &
    eval_pid=$!
    wait "${eval_pid}"
    status=$?
    cleanup_process_group "${eval_pid}"

    if [[ ${status} -eq 0 ]]; then
        echo "[INFO] Finished ${run_name} (${env_config_name})"
        status_text="ok"
    elif [[ ${status} -eq 124 || ${status} -eq 137 ]]; then
        echo "[ERROR] Evaluation timed out or was killed for ${run_name} after ${EVAL_TIMEOUT}" >&2
        status_text="timeout_or_killed:${status}"
        overall_status=${status}
    else
        echo "[ERROR] Evaluation failed for ${run_name} with exit code ${status}" >&2
        status_text="failed:${status}"
        overall_status=${status}
    fi

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${CONFIG_NAME}" "${CONFIG}" "${env_config_name}" "${env_config}" \
        "${MODEL_NAME_RESOLVED}" "${MODEL_DIR}" "${run_name}" "${CHECKPOINT}" \
        "${TARGET_EPISODES_VALUE}" "${NUM_ENVS_VALUE}" "${status_text}" "${log_file}" >> "${SUMMARY_FILE}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "${CONFIG_NAME}" "${CONFIG}" "${env_config_name}" "${env_config}" \
        "${MODEL_NAME_RESOLVED}" "${MODEL_DIR}" "${run_name}" "${CHECKPOINT}" \
        "${TARGET_EPISODES_VALUE}" "${NUM_ENVS_VALUE}" "${status_text}" "${log_file}" >> "${CONFIG_SUMMARY_FILE}"

    echo "[INFO] Sleeping ${SLEEP_BETWEEN_RUNS}s before the next evaluation to let Isaac/driver resources settle."
    sleep "${SLEEP_BETWEEN_RUNS}"
done

echo "[INFO] Evaluation sweep complete. Status summary: ${SUMMARY_FILE}"
exit "${overall_status}"
