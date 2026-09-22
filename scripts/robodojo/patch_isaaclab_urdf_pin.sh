#!/usr/bin/env bash
# Relax IsaacLab's exact pin on the Isaac Sim URDF importer extension.
#
# Why this is needed
#   RoboDojo/third_party/IsaacLab/apps/isaaclab.python.kit pins
#     "isaacsim.asset.importer.urdf" = {version = "2.4.31", exact = true}
#   but `isaacsim[all,extscache]==5.1.0` - the version RoboDojo's own
#   scripts/install.sh installs - ships 2.4.30. Kit would normally download the
#   pinned build from the NVIDIA extension registry, but the pip install is
#   registry-less ("Syncing with extension registry unavailable"), so the
#   dependency solver fails and Isaac Sim exits before the sim client starts:
#     Failed to resolve extension dependencies ... can't be satisfied
#     ModuleNotFoundError: No module named 'omni.kit.usd'
#
# Why relaxing it is safe here
#   The pin exists only to keep the old fixed-joint merging behaviour when
#   CONVERTING URDF to USD (IsaacLab CHANGELOG.rst:340). RoboDojo never runs
#   the importer: robots are spawned from prebuilt USD (Assets/Robots/x5/
#   configuration/ARX_*.usd) and the .urdf is parsed by cuRobo's own loader for
#   IK. Nothing in env/ or utils/ touches isaaclab.sim.converters.
#
# The proper fix belongs upstream (align the IsaacLab submodule pin with the
# Isaac Sim version RoboDojo installs). This is a local, reversible unblock.
#
# Usage: bash scripts/robodojo/patch_isaaclab_urdf_pin.sh [--revert]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KIT_FILE="${REPO_ROOT}/RoboDojo/third_party/IsaacLab/apps/isaaclab.python.kit"
BACKUP="${KIT_FILE}.robodojo-vlm.orig"
EXT_NAME="isaacsim.asset.importer.urdf"

fail() { echo "[patch_isaaclab_urdf_pin] $*" >&2; exit 1; }

[[ -f "${KIT_FILE}" ]] || fail "kit experience file not found: ${KIT_FILE}"

if [[ "${1:-}" == "--revert" ]]; then
    [[ -f "${BACKUP}" ]] || fail "nothing to revert: ${BACKUP} does not exist"
    mv "${BACKUP}" "${KIT_FILE}"
    echo "[patch_isaaclab_urdf_pin] reverted ${KIT_FILE}"
    exit 0
fi

# Patch to whatever version is actually installed rather than a hardcoded one,
# so this keeps working across Isaac Sim upgrades.
installed_version="$(python - <<'PY'
import os, sys, sysconfig
toml = os.path.join(sysconfig.get_paths()["purelib"], "isaacsim", "exts",
                    "isaacsim.asset.importer.urdf", "config", "extension.toml")
if not os.path.isfile(toml):
    sys.exit(0)
for line in open(toml, encoding="utf-8"):
    line = line.strip()
    if line.startswith("version"):
        print(line.split("=", 1)[1].strip().strip('"'))
        break
PY
)"

[[ -n "${installed_version}" ]] || fail "could not read the installed ${EXT_NAME} version; is the Isaac Sim conda env active?"

current_pin="$(grep -oP "(?<=\"${EXT_NAME}\" = \{version = \")[^\"]+" "${KIT_FILE}" || true)"
[[ -n "${current_pin}" ]] || fail "no ${EXT_NAME} pin found in ${KIT_FILE}"

if [[ "${current_pin}" == "${installed_version}" ]]; then
    echo "[patch_isaaclab_urdf_pin] already aligned: pin=${current_pin}, installed=${installed_version}"
    exit 0
fi

[[ -f "${BACKUP}" ]] || cp "${KIT_FILE}" "${BACKUP}"
sed -i "s/\"${EXT_NAME}\" = {version = \"${current_pin}\"/\"${EXT_NAME}\" = {version = \"${installed_version}\"/" "${KIT_FILE}"

echo "[patch_isaaclab_urdf_pin] ${EXT_NAME}: pin ${current_pin} -> ${installed_version}"
echo "[patch_isaaclab_urdf_pin] backup at ${BACKUP} (restore with --revert)"
grep -n "${EXT_NAME}" "${KIT_FILE}"
