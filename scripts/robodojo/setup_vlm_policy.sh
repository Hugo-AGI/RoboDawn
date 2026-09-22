#!/usr/bin/env bash
# Mount the VLM policy into RoboDojo and check its prerequisites.
#
# The policy code lives in this repository (evaluation/policies/vlm_agent) but
# RoboDojo resolves policies as XPolicyLab/policy/<NAME>, so it is symlinked
# into place rather than committed into the submodule.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
POLICY_NAME="vlm_agent"
SOURCE_DIR="${REPO_ROOT}/evaluation/policies/${POLICY_NAME}"
POLICY_PARENT="${REPO_ROOT}/RoboDojo/XPolicyLab/policy"
LINK_PATH="${POLICY_PARENT}/${POLICY_NAME}"
POLICY_ENV="${POLICY_ENV:-vlm_policy}"

fail() { echo "[setup_vlm_policy] $*" >&2; exit 1; }

[[ -d "${SOURCE_DIR}" ]] || fail "policy source not found: ${SOURCE_DIR}"
[[ -d "${POLICY_PARENT}" ]] || fail "XPolicyLab submodule not initialised: ${POLICY_PARENT}"

if [[ -e "${LINK_PATH}" && ! -L "${LINK_PATH}" ]]; then
    fail "${LINK_PATH} exists and is not a symlink; remove it before rerunning"
fi
ln -sfn "../../../evaluation/policies/${POLICY_NAME}" "${LINK_PATH}"
echo "[setup_vlm_policy] linked ${LINK_PATH} -> $(readlink "${LINK_PATH}")"

for required in deploy.py deploy.yml model.py motion.py curobo_motion.py commands.py main_route.py main_prompts.py profiles/robodojo_x5_main.yaml setup_eval_policy_server.sh setup_eval_env_client.sh eval.sh; do
    [[ -f "${LINK_PATH}/${required}" ]] || fail "missing ${required} behind the symlink"
done

if [[ -f "${REPO_ROOT}/secrets.json" ]]; then
    echo "[setup_vlm_policy] secrets.json found at ${REPO_ROOT}/secrets.json"
else
    echo "[setup_vlm_policy] WARNING: no secrets.json at ${REPO_ROOT}; the policy server will not start." >&2
    echo "                   Use secrets.example.json: {\"vlm\": [{\"name\": ..., \"model\": ..., \"base_url\": ..., \"api_key\": ...}]}" >&2
fi

# The asset bundle ships robot configs rendered with the packager's absolute
# paths, which makes cuRobo fail during env construction. Repair is idempotent,
# and skipped with a warning when assets are not downloaded yet.
if [[ -d "${REPO_ROOT}/RoboDojo/Assets/Robots" ]]; then
    bash "${REPO_ROOT}/scripts/robodojo/fix_asset_paths.sh"
else
    echo "[setup_vlm_policy] assets not downloaded yet; run scripts/robodojo/fix_asset_paths.sh after RoboDojo's init_assets.sh" >&2
fi

# The policy server needs websockets>=13 (websockets.asyncio), which the Isaac
# Sim env deliberately does not have - hence a separate policy env.
if conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${POLICY_ENV}"; then
    echo "[setup_vlm_policy] policy conda env '${POLICY_ENV}' found"
else
    cat >&2 <<EOF
[setup_vlm_policy] WARNING: policy conda env '${POLICY_ENV}' not found. Create it with:

  conda create -n ${POLICY_ENV} python=3.11 -y && conda activate ${POLICY_ENV} && \\
  pip install "numpy>=1.23,<2" "pyyaml>=6" "opencv-python-headless>=4.8" "h5py>=3.8" \\
              "websockets>=14" "msgpack>=1.0.8" "msgpack-numpy>=0.4.8" "pydantic>=2.5" "pillow>=10" \\
              -r ${SOURCE_DIR}/requirements.txt
EOF
fi

echo "[setup_vlm_policy] done. Run an eval with: bash scripts/robodojo/run_vlm_eval.sh --task stack_bowls"
