#!/usr/bin/env bash
# Re-render RoboDojo's robot configs for THIS machine's asset location.
#
# Why this is needed
#   Assets/Robots/<robot>/curobo.yml is shipped inside the downloaded asset
#   bundle already rendered on whoever packaged it. It therefore carries their
#   absolute paths:
#       urdf_path: /home/dong/RoboDojo/Assets/Robots/x5/X5A.urdf
#   cuRobo loads that path directly while building the IK/motion planner, so
#   every eval dies during env construction with:
#       ValueError: /home/dong/RoboDojo/Assets/Robots/x5/X5A.urdf is not a file
#   Every machine that downloads the bundle hits this, not just a broken install.
#
# The fix is RoboDojo's own tool: utils/update_embodiment_config_path.py renders
# the *_tmp.yml templates (which use ${ASSETS_PATH}) into the real *.yml using
# the current checkout's path. This wrapper makes that step automatic and
# idempotent instead of a manual one that is easy to forget - re-downloading or
# re-initialising assets silently reintroduces the packaged paths.
#
# Usage: bash scripts/robodojo/fix_asset_paths.sh [--check]
#   --check  report only, exit 1 if a re-render is needed (for CI / doctor use)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROBODOJO_ROOT="${REPO_ROOT}/RoboDojo"
RENDERER="${ROBODOJO_ROOT}/utils/update_embodiment_config_path.py"
CHECK_ONLY="false"
[[ "${1:-}" == "--check" ]] && CHECK_ONLY="true"

fail() { echo "[fix_asset_paths] $*" >&2; exit 1; }

[[ -d "${ROBODOJO_ROOT}/Assets/Robots" ]] || fail "assets not initialised: ${ROBODOJO_ROOT}/Assets/Robots not found"
[[ -f "${RENDERER}" ]] || fail "RoboDojo path renderer not found: ${RENDERER}"

# Report every filesystem path referenced by a robot config that does not exist.
# Stdlib only, so this runs with or without a conda env active.
report_broken() {
    python3 - "${ROBODOJO_ROOT}" <<'PY'
import glob, os, sys

root = sys.argv[1]
broken = []
for path in sorted(glob.glob(os.path.join(root, "Assets", "Robots", "*", "*.yml"))):
    if path.endswith("_tmp.yml"):
        continue
    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            key, sep, value = line.partition(":")
            if not sep or not key.strip().endswith(("urdf_path", "asset_root_path")):
                continue
            value = value.strip().strip('"').strip("'")
            # Unrendered templates and nulls are not stale paths.
            if not value or value in ("null", "~") or "${" in value or "$ASSETS_PATH" in value:
                continue
            # Only absolute paths can be stale: relative ones (robot_config.yml's
            # "./X5A.urdf") are resolved by the loader against the robot dir.
            if not value.startswith("/"):
                continue
            if not os.path.exists(value):
                broken.append(f"{os.path.relpath(path, root)}:{lineno}: {value}")
print("\n".join(broken))
PY
}

broken="$(report_broken)"

if [[ -z "${broken}" ]]; then
    echo "[fix_asset_paths] robot configs already point at this machine's assets; nothing to do"
    exit 0
fi

echo "[fix_asset_paths] stale absolute paths found in the asset bundle:"
echo "${broken}" | sed 's/^/    /'

if [[ "${CHECK_ONLY}" == "true" ]]; then
    echo "[fix_asset_paths] --check: run without --check to re-render" >&2
    exit 1
fi

# The renderer resolves ASSETS_PATH from its working directory and prompts on
# stdin if it cannot find Assets/Robots; we checked that above, so close stdin
# to keep this non-interactive.
echo "[fix_asset_paths] re-rendering with ${RENDERER}"
(cd "${ROBODOJO_ROOT}" && python3 "${RENDERER}" < /dev/null > /dev/null) || fail "renderer failed"

still_broken="$(report_broken)"
if [[ -n "${still_broken}" ]]; then
    echo "[fix_asset_paths] still broken after re-rendering:" >&2
    echo "${still_broken}" | sed 's/^/    /' >&2
    fail "the *_tmp.yml templates may be missing or stale"
fi

echo "[fix_asset_paths] repaired; all robot config paths now resolve"
