"""Bootstrap patched pip through refreshed OS roots on Python 3.10 or newer."""

import os
import re
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

PIP_TARGET = "26.2.2"
CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"


def bootstrap(pip_version=None, environment=None, runner=None, bundle=CA_BUNDLE):
    """Handle old pip without enabling newer pip's additive vendored roots."""
    if not Path(bundle).is_file() or Path(bundle).stat().st_size == 0:
        raise RuntimeError("Refreshed OS CA bundle is missing or empty")
    selected_version = pip_version if pip_version is not None else version("pip")
    match = re.match(r"^(\d+)\.(\d+)", selected_version)
    if match is None:
        raise RuntimeError("Cannot determine initial pip certificate behavior")
    initial = tuple(map(int, match.groups()))
    env = dict(os.environ if environment is None else environment)
    for name in (
        "PIP_USE_DEPRECATED",
        "PIP_USE_FEATURE",
        "PIP_TRUSTED_HOST",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
    ):
        env.pop(name, None)
    env["PIP_CONFIG_FILE"] = os.devnull
    env["PIP_CERT"] = bundle
    if initial >= (24, 2):
        env["PIP_USE_DEPRECATED"] = "legacy-certs"
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--no-cache-dir",
        "--only-binary=:all:",
        "--no-deps",
        "pip==" + PIP_TARGET,
    ]
    execute = subprocess.run if runner is None else runner
    execute(command, env=env, check=True)


if __name__ == "__main__":
    bootstrap()
