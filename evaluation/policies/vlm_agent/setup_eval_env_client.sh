#!/bin/bash
# Start the RoboDojo simulation client against an already-running VLM policy
# server. Same contract as every other XPolicyLab policy.
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
env_gpu_id=${7}
eval_env_conda_env=${8}
additional_info=${9}
policy_server_port=${10}
policy_server_ip=${11:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"
UTILS_DIR="${XPL_ROOT}/utils"

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[CLIENT] policy=${policy_name}, task=${task_name}, server=${policy_server_ip}:${policy_server_port}"

prefix=()
if [[ "${EVAL_ENV_TYPE:-sim}" == "sim" || -z "${EVAL_ENV_TYPE:-}" ]]; then
    SOURCE_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
    if ! command -v ffmpeg >/dev/null 2>&1; then
        TOOLS_BIN="${SOURCE_DIR}/../../../.cache/robodojo-tools/bin"
        if [[ ! -x "${TOOLS_BIN}/ffmpeg" ]]; then
            echo "[CLIENT] ffmpeg is required for simulation; add it to PATH or .cache/robodojo-tools/bin" >&2
            exit 1
        fi
        export PATH="${TOOLS_BIN}:${PATH}"
    fi
    COMPAT_SCRIPT="${SOURCE_DIR}/../../../scripts/robodojo/compat/vulkan_driver_compat.py"
    prefix=(python3 "${COMPAT_SCRIPT}" --isaac51 --gpu-index "${env_gpu_id}" \
            --mode "${ROBODOJO_VULKAN_COMPAT:-auto}" --)
fi

exec "${prefix[@]}" bash "${UTILS_DIR}/setup_env_client.sh" \
    "${UTILS_DIR}" \
    "${yaml_file}" \
    "${eval_env_conda_env}" \
    "${policy_server_port}" \
    "${bench_name}" \
    "${task_name}" \
    "${env_cfg_type}" \
    "${policy_name}" \
    "${additional_info}" \
    "${BENCH_ROOT}" \
    "${seed}" \
    "${env_gpu_id}" \
    "${policy_server_ip}"
