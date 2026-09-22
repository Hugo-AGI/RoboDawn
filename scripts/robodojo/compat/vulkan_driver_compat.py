"""Run one command with the verified NVIDIA 595 / Isaac Sim 5.1 workaround."""

from __future__ import annotations

import argparse
import ctypes
from datetime import date
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from vulkan_probe import probe


REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE = REPO_ROOT / ".cache/robodojo-vulkan"
PACKAGE_NAME = "vulkan-profiles_1.4.313.0~rc1-1lunarg22.04-1_amd64.deb"
PACKAGE_URL = "https://packages.lunarg.com/vulkan/pool/main/v/vulkan-profiles/" + PACKAGE_NAME
PACKAGE_SHA256 = "e126f4ea78a3d7a5fb9a3976ff1da12647854455959e26a6e34f60b4410087c9"
LAYER_SHA256 = "9699dce1cae0e3eabc751af596858ec5aade20f40584509ee23a19f2b995a0f3"
LAYER = CACHE / "libVkLayer_khronos_profiles.so"
JSONCPP_PACKAGE_NAME = "libjsoncpp25_1.9.5-3_amd64.deb"
JSONCPP_URL = "https://archive.ubuntu.com/ubuntu/pool/main/libj/libjsoncpp/" + JSONCPP_PACKAGE_NAME
JSONCPP_PACKAGE_SHA256 = "8844bfab4691b56cf49e741b65389147b03168bef4254b31ae97ad679d199a78"
JSONCPP_SHA256 = "7d699962170ac3142f81ed55df0bcdd49d035c0e3374378d49020435b68f68d9"
JSONCPP = CACHE / "deps/libjsoncpp.so.25"
CAP = 4 * 1024**3 - 2 * 1024**2
PROFILE_NAME = "VP_ROBODOJO_nvidia_595_compat"
MANIFEST_DIR = CACHE / "xdg-data/vulkan/implicit_layer.d"
ENABLE = "ROBODOJO_VULKAN_COMPAT_ENABLE_LAYER"
DISABLE = "ROBODOJO_VULKAN_COMPAT_DISABLE_LAYER"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hashlib.sha256(stream.read()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def package_archive(name: str, url: str, checksum: str, supplied: Path | None = None) -> tarfile.TarFile:
    local_package = CACHE / name
    if supplied is not None:
        if digest(supplied) != checksum:
            raise RuntimeError(f"{name}: package SHA256 mismatch")
        if supplied.resolve() != local_package.resolve():
            shutil.copyfile(supplied, local_package)
    if not local_package.is_file():
        temporary = None
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                with tempfile.NamedTemporaryFile(dir=CACHE, delete=False) as output:
                    temporary = Path(output.name)
                    shutil.copyfileobj(response, output)
            if digest(temporary) != checksum:
                raise RuntimeError(f"{name}: downloaded package SHA256 mismatch")
            temporary.replace(local_package)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if digest(local_package) != checksum:
        raise RuntimeError(f"{name}: cached package SHA256 mismatch")
    archive_bytes = subprocess.run(
        ["dpkg-deb", "--fsys-tarfile", str(local_package)], capture_output=True, check=True,
    ).stdout
    return tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:")


def extract_member(archive: tarfile.TarFile, member_name: str, target: Path) -> None:
    member = archive.getmember(member_name)
    if not member.isfile():
        raise RuntimeError("unexpected package member type")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(archive.extractfile(member).read())


def prepare(package: Path | None = None, jsoncpp_package: Path | None = None) -> None:
    """Extract verified official binaries locally; execute no package scripts."""
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / ".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with package_archive(PACKAGE_NAME, PACKAGE_URL, PACKAGE_SHA256, package) as archive:
            for member_name, target in (
                ("./usr/lib/x86_64-linux-gnu/libVkLayer_khronos_profiles.so", LAYER),
                ("./usr/share/doc/vulkan-profiles/copyright", CACHE / "COPYRIGHT"),
            ):
                extract_member(archive, member_name, target)
        if digest(LAYER) != LAYER_SHA256:
            raise RuntimeError("extracted layer SHA256 mismatch")
        with package_archive(JSONCPP_PACKAGE_NAME, JSONCPP_URL, JSONCPP_PACKAGE_SHA256, jsoncpp_package) as archive:
            extract_member(archive, "./usr/lib/x86_64-linux-gnu/libjsoncpp.so.1.9.5", JSONCPP)
            extract_member(archive, "./usr/share/doc/libjsoncpp25/copyright", CACHE / "COPYRIGHT.jsoncpp")
        if digest(JSONCPP) != JSONCPP_SHA256:
            raise RuntimeError("extracted jsoncpp SHA256 mismatch")
        profile_dir = CACHE / "profiles"
        manifest_dir = MANIFEST_DIR
        write_json(profile_dir / f"{PROFILE_NAME}.json", {
            "$schema": "https://schema.khronos.org/vulkan/profiles-0.8-latest.json#",
            "capabilities": {"allocation_limit": {"properties": {
                "VkPhysicalDeviceMaintenance3Properties": {"maxMemoryAllocationSize": CAP},
            }}},
            "profiles": {PROFILE_NAME: {
                "version": 1, "api-version": "1.1.0", "label": "RoboDojo NVIDIA 595 compatibility",
                "description": "Limit one reported memory allocation property for Isaac Sim 5.1.",
                "contributors": {"local": {"github": "https://github.com/KhronosGroup/Vulkan-Profiles"}},
                "history": [{"revision": 1, "date": date.today().isoformat(), "author": "local", "comment": "Process-local allocation limit."}],
                "capabilities": ["allocation_limit"],
            }},
        })
        write_json(manifest_dir / "VkLayer_khronos_profiles.json", {
            "file_format_version": "1.2.1",
            "layer": {
                "name": "VK_LAYER_KHRONOS_profiles", "type": "GLOBAL",
                "library_path": str(LAYER), "api_version": "1.4.313",
                "implementation_version": "1", "description": "RoboDojo-local Khronos Profiles layer",
                "enable_environment": {ENABLE: "1"}, "disable_environment": {DISABLE: "1"},
            },
        })
        write_json(CACHE / "package_info.json", {
            "url": PACKAGE_URL, "package_sha256": PACKAGE_SHA256,
            "layer_sha256": LAYER_SHA256, "maxMemoryAllocationSize": CAP,
            "jsoncpp_url": JSONCPP_URL, "jsoncpp_package_sha256": JSONCPP_PACKAGE_SHA256,
            "jsoncpp_sha256": JSONCPP_SHA256,
            "issue": "https://github.com/isaac-sim/IsaacSim/issues/568#issuecomment-4415180467",
        })


def environment(mode: str, report: dict, original: dict) -> tuple[dict, bool]:
    child = dict(original)
    needs_compat = any(
        device["driver"][0] == 595 and device["maxMemoryAllocationSize"] == (1 << 64) - 1
        for device in report["devices"]
    )
    if mode == "off" or not needs_compat:
        child.pop(ENABLE, None)
        child[DISABLE] = "1"
        return child, False
    if any(device.get("device_type") != 4 for device in report.get("other_devices", [])):
        raise RuntimeError("compatibility profile is limited to NVIDIA GPUs plus CPU software Vulkan; another physical vendor is present")
    if not all(device["driver"][0] == 595 and device["maxMemoryAllocationSize"] == (1 << 64) - 1 for device in report["devices"]):
        raise RuntimeError("compatibility profile requires the same affected NVIDIA 595 limit on every physical GPU")
    if not LAYER.is_file() or digest(LAYER) != LAYER_SHA256:
        raise RuntimeError("compatibility layer missing or changed; run vulkan_driver_compat.py --prepare first")
    if not JSONCPP.is_file() or digest(JSONCPP) != JSONCPP_SHA256:
        raise RuntimeError("compatibility JSON dependency missing or changed; run vulkan_driver_compat.py --prepare first")
    for path in (CACHE / "profiles" / f"{PROFILE_NAME}.json", MANIFEST_DIR / "VkLayer_khronos_profiles.json"):
        if not path.is_file():
            raise RuntimeError("compatibility configuration missing; run vulkan_driver_compat.py --prepare first")
    # Resolve dependencies now, so a failed layer load cannot fall through to a
    # known crashing renderer startup. This does not initialize the layer.
    ctypes.CDLL(str(JSONCPP), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(LAYER))
    child.pop(DISABLE, None)
    child.update({
        ENABLE: "1",
        "VK_KHRONOS_PROFILES_PROFILE_NAME": PROFILE_NAME,
        "VK_KHRONOS_PROFILES_PROFILE_DIRS": str(CACHE / "profiles"),
        "VK_KHRONOS_PROFILES_SIMULATE_CAPABILITIES": "SIMULATE_PROPERTIES_BIT",
        "VK_KHRONOS_PROFILES_EMULATE_PORTABILITY": "false",
        "VK_KHRONOS_PROFILES_DEBUG_REPORTS": "DEBUG_REPORT_ERROR_BIT",
        # The layer also enumerates unused CPU software Vulkan devices, whose
        # limit can be 2 GiB. It cannot filter property overrides by vendor.
        "VK_KHRONOS_PROFILES_DEBUG_FAIL_ON_ERROR": "false",
    })
    manifest_dir = str(MANIFEST_DIR)
    old_path = child.get("VK_ADD_IMPLICIT_LAYER_PATH")
    child["VK_ADD_IMPLICIT_LAYER_PATH"] = manifest_dir + (os.pathsep + old_path if old_path else "")
    # Ubuntu 22.04 loaders can predate VK_ADD_IMPLICIT_LAYER_PATH support.
    child["XDG_DATA_DIRS"] = str(CACHE / "xdg-data") + os.pathsep + child.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    old_libraries = child.get("LD_LIBRARY_PATH")
    child["LD_LIBRARY_PATH"] = str(JSONCPP.parent) + (os.pathsep + old_libraries if old_libraries else "")
    return child, True


def validate_override(before: dict, child: dict) -> None:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe"],
        env=child, capture_output=True, text=True, timeout=60, check=True,
    )
    after = json.loads(completed.stdout)
    if len(after["devices"]) != len(before["devices"]):
        raise RuntimeError("compatibility verification changed the NVIDIA device count")
    for original, overridden in zip(before["devices"], after["devices"]):
        expected = dict(original, maxMemoryAllocationSize=CAP)
        if overridden != expected:
            raise RuntimeError("compatibility verification did not change only the intended NVIDIA allocation property")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("auto", "off"), default=os.environ.get("ROBODOJO_VULKAN_COMPAT", "auto"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--package", type=Path, help="use an already downloaded, checksum-matched official .deb")
    parser.add_argument("--jsoncpp-package", type=Path, help="use an already downloaded, checksum-matched Ubuntu dependency")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--isaac51", action="store_true", help="reject known unsupported old renderer drivers before startup")
    parser.add_argument("--gpu-index", type=int, default=0, help="NVIDIA GPU ordinal selected by the child renderer")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.package, args.jsoncpp_package)
        print(f"[vulkan_compat] prepared {LAYER}")
        return 0
    if args.package or args.jsoncpp_package:
        parser.error("package arguments require --prepare")
    if args.probe:
        print(json.dumps(probe(), indent=2))
        return 0
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required after --")
    report = probe() if args.mode != "off" else {"devices": []}
    if args.isaac51 and args.mode != "off" and not 0 <= args.gpu_index < len(report["devices"]):
        raise RuntimeError("selected renderer GPU is not a visible NVIDIA Vulkan device")
    if args.isaac51 and args.mode != "off" and any(
        tuple(device["driver"][:2]) <= (570, 144) for device in report["devices"]
    ):
        raise RuntimeError(
            "Isaac Sim 5.1 renderer requires a newer driver than this host provides; "
            "use a host with a supported driver. --mode off explicitly bypasses this check for diagnosis."
        )
    child, active = environment(args.mode, report, os.environ)
    if active:
        validate_override(report, child)
        print(f"[vulkan_compat] NVIDIA 595 allocation limit: UINT64_MAX -> {CAP}", file=sys.stderr, flush=True)
    os.execvpe(command[0], command, child)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[vulkan_compat] {exc}", file=sys.stderr)
        raise SystemExit(2)
