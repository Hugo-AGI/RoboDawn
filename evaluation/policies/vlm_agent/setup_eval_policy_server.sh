#!/bin/bash
# Start the VLM policy server. Called by RoboDojo's run_policy_eval.sh with
# CWD set to this directory (which is XPolicyLab/policy/vlm_agent, a symlink
# into the benchmark repo).
set -euo pipefail

bench_name=${1}
task_name=${2}
ckpt_name=${3}
env_cfg_type=${4}
action_type=${5}
seed=${6}
policy_gpu_id=${7}
policy_conda_env=${8}
policy_server_port=${9}
policy_server_host=${10:-"localhost"}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
XPL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BENCH_ROOT="$(cd "${XPL_ROOT}/.." && pwd)"

if [[ ! -f "${XPL_ROOT}/setup_policy_server.py" ]]; then
    echo "[SERVER] expected XPolicyLab root at ${XPL_ROOT}; run this policy from" >&2
    echo "         RoboDojo/XPolicyLab/policy/vlm_agent (see scripts/robodojo/setup_vlm_policy.sh)" >&2
    exit 1
fi

policy_name="$(basename "${SCRIPT_DIR}")"
yaml_file="${SCRIPT_DIR}/deploy.yml"

echo "[SERVER] policy=${policy_name}, task=${task_name}, policy_server_port=${policy_server_port}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${policy_conda_env}"

# XPolicyLab is not pip-installed in the eval envs, so put both roots on the
# path: BENCH_ROOT for `XPolicyLab.*`, XPL_ROOT for `client_server.*`.
exec env \
    PYTHONWARNINGS=ignore::UserWarning \
    PYTHONPATH="${BENCH_ROOT}:${XPL_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="${policy_gpu_id}" \
    python "${XPL_ROOT}/setup_policy_server.py" \
        --config_path "${yaml_file}" \
        --overrides \
            port="${policy_server_port}" \
            host="${policy_server_host}" \
            bench_name="${bench_name}" \
            task_name="${task_name}" \
            ckpt_name="${ckpt_name}" \
            env_cfg_type="${env_cfg_type}" \
            seed="${seed}" \
            policy_name="${policy_name}" \
            action_type="${action_type}"
