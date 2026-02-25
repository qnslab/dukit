#!/usr/bin/env python3
"""Install cpufit/gpufit wheels based on platform.

Usage:
    python install_gpufit_wheels.py cpufit
    python install_gpufit_wheels.py gpufit
    python install_gpufit_wheels.py both
"""

import sys
import subprocess
import platform
from pathlib import Path

WHEEL_DIR = Path(__file__).parent / "gpufit_wheels"

WHEELS = {
    "cpufit": {
        "Linux": "pyCpufit-101.2.0-py2.py3-none-linux_x86_64.whl",
        "Windows": "pyCpufit-101.2.0-py2.py3-none-win_amd64.whl",
    },
    "gpufit": {
        "Windows": "pyGpufit-101.2.0-py2.py3-none-win_amd64.whl",
    },
}


def get_platform():
    system = platform.system()
    if system == "Linux":
        return "Linux"
    elif system == "Windows":
        return "Windows"
    else:
        raise RuntimeError(f"Unsupported platform: {system}")


def install_wheel(package_type: str):
    plat = get_platform()
    wheels = WHEELS.get(package_type)
    if not wheels:
        print(f"Unknown package type: {package_type}", file=sys.stderr)
        sys.exit(1)

    wheel_name = wheels.get(plat)
    if not wheel_name:
        print(f"{package_type} not available on {plat}", file=sys.stderr)
        sys.exit(1)

    wheel_path = WHEEL_DIR / wheel_name
    if not wheel_path.exists():
        print(f"Wheel not found: {wheel_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Installing {package_type} from {wheel_path}...")
    subprocess.run(["uv", "pip", "install", str(wheel_path)], check=True)
    print(f"{package_type} installed successfully")


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    target = sys.argv[1].lower()

    if target == "cpufit":
        install_wheel("cpufit")
    elif target == "gpufit":
        install_wheel("gpufit")
    elif target == "both":
        install_wheel("cpufit")
        try:
            install_wheel("gpufit")
        except SystemExit:
            print("Note: gpufit not available on this platform (Windows only)")
    else:
        print(__doc__, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
