#!/usr/bin/env bash
# Run one RoboDojo task with the VLM policy (policy server + Isaac Sim client).
#
# Usage:
#   bash scripts/robodojo/run_vlm_eval.sh [--task NAME] [--eval-num N] [--seed N]
#                                         [--model NAME | --list-models]
#                                         [--env-gpu ID] [--policy-gpu ID]
#                                         [--policy-env ENV] [--eval-env ENV]
#                                         [--ckpt NAME] [-- <extra robodojo.sh args>]
#
# Every knob of the policy comes from evaluation/policies/vlm_agent/deploy.yml,
# optionally overridden by a JSON object in VLM_AGENT_OVERRIDES (this is how
# scripts/robodojo/run_vlm_experiment.py drives it).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_ROOT="${REPO_ROOT}/RoboDojo"

task="stack_bowls"
eval_num="5"
seed="0"
env_gpu="0"
policy_gpu="0"
policy_env="${POLICY_ENV:-vlm_policy}"
eval_env="${EVAL_ENV:-RoboDojo}"
ckpt="vlm"
vlm_model=""
list_models="false"
extra=()

need_value() {
    if [[ $# -lt 2 || -z "$2" || "$2" == --* ]]; then
        echo "[run_vlm_eval] $1 requires a value" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model|--profile) need_value "$@"; vlm_model="$2"; shift 2 ;;
        --list-models) list_models="true"; shift ;;
        --task) task="$2"; shift 2 ;;
        --eval-num) eval_num="$2"; shift 2 ;;
        --seed) seed="$2"; shift 2 ;;
        --env-gpu) env_gpu="$2"; shift 2 ;;
        --policy-gpu) policy_gpu="$2"; shift 2 ;;
        --policy-env) policy_env="$2"; shift 2 ;;
        --eval-env) eval_env="$2"; shift 2 ;;
        --ckpt) ckpt="$2"; shift 2 ;;
        --) shift; extra=("$@"); break ;;
        -h|--help) sed -n '2,11p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "[run_vlm_eval] unknown argument: $1" >&2; exit 2 ;;
    esac
done

PROFILE_CLIENT="${REPO_ROOT}/evaluation/policies/vlm_agent/vlm_client.py"
if [[ "${list_models}" == "true" ]]; then
    exec python3 "${PROFILE_CLIENT}" --list-models
fi
if [[ -n "${vlm_model}" ]]; then
    export VLM_AGENT_PROFILE="${vlm_model}"
fi
if [[ -n "${VLM_AGENT_PROFILE:-}" ]]; then
    resolved_profile="$(python3 "${PROFILE_CLIENT}" --model "${VLM_AGENT_PROFILE}")"
    export VLM_AGENT_PROFILE="${resolved_profile}"
    echo "[run_vlm_eval] model profile=${VLM_AGENT_PROFILE}"
fi

# Isaac Sim otherwise blocks on an interactive EULA prompt and dies with
# "Unable to bootstrap inner kit kernel: EOF when reading a line". RoboDojo sets
# the same variables in its own Dockerfile and scripts/install.sh.
export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"

# XPolicyLab's setup_env_client.sh parses deploy.yml with a bare `python`
# before it activates any conda env, so the launching shell needs a python that
# has pyyaml. Activate the simulator env up front: it has one, and it is the env
# the sim client ends up using anyway.
if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${eval_env}"
    echo "[run_vlm_eval] launcher python: $(command -v python)"
else
    echo "[run_vlm_eval] WARNING: conda not on PATH; the sim client launcher needs a python with pyyaml" >&2
fi

# The policy emits planned joint trajectories, so RoboDojo labels the result
# directory "joint". VLM_AGENT_OVERRIDES, when set, must be a JSON object; a
# bad value fails here instead of mid-episode. Only python3 is used: this block
# also runs in the launcher tests, which provide neither conda nor coreutils.
action_type="joint"
if [[ -n "${VLM_AGENT_OVERRIDES:-}" ]]; then
    if ! python3 -c 'import json, os, sys; sys.exit(0 if isinstance(json.loads(os.environ["VLM_AGENT_OVERRIDES"]), dict) else 1)' 2>/dev/null; then
        echo "[run_vlm_eval] VLM_AGENT_OVERRIDES must be a JSON object" >&2
        exit 2
    fi
fi

if [[ ! -e "${ROBODOJO_ROOT}/XPolicyLab/policy/vlm_agent" ]]; then
    echo "[run_vlm_eval] policy not linked yet; running setup_vlm_policy.sh first"
    bash "${REPO_ROOT}/scripts/robodojo/setup_vlm_policy.sh"
fi

echo "[run_vlm_eval] task=${task} eval_num=${eval_num} seed=${seed}"
echo "[run_vlm_eval] policy_env=${policy_env} eval_env=${eval_env} env_gpu=${env_gpu} action_type=${action_type}"

exec bash "${ROBODOJO_ROOT}/scripts/robodojo.sh" eval \
    --policy-dir XPolicyLab/policy/vlm_agent \
    --task "${task}" \
    --ckpt "${ckpt}" \
    --policy-env "${policy_env}" \
    --eval-env "${eval_env}" \
    --action-type "${action_type}" \
    --seed "${seed}" \
    --env-gpu "${env_gpu}" \
    --policy-gpu "${policy_gpu}" \
    --eval-num "${eval_num}" \
    ${extra[@]+"${extra[@]}"}
