#!/usr/bin/env python3
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "preflight_dgxspark.sh"


def make_stub(directory: pathlib.Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def make_env(path: pathlib.Path, expected_ip: str = "10.0.0.1") -> None:
    path.write_text(
        "\n".join(
            [
                f"HEAD_IP={expected_ip}",
                "HEAD_CX7_IF=lo",
                "HEAD_CX7_IB=nonexistent_test_rdma",
                "NCCL_IB_GID_INDEX=3",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_preflight(bins: pathlib.Path, envfile: pathlib.Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PATH"] = f"{bins}:/usr/bin:/bin"
    return subprocess.run(
        ["/bin/bash", str(SCRIPT), "head", str(envfile)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ninfer-preflight-tests-") as td:
        root = pathlib.Path(td)
        bins = root / "bin"
        bins.mkdir()
        envfile = root / "contract.env"
        make_env(envfile)

        make_stub(bins, "ip", 'printf "lo UNKNOWN 110.0.0.1/8\\n"\n')
        make_stub(bins, "nvidia-smi", 'printf "NVIDIA GB10, 0, 0\\n"\n')
        make_stub(bins, "ethtool", 'printf "Speed: 200000Mb/s\\nLink detected: yes\\n"\n')
        run = run_preflight(bins, envfile)
        assert run.returncode == 2
        assert "FAIL  lo does not own expected IP 10.0.0.1" in run.stdout
        assert "OK    lo owns 10.0.0.1" not in run.stdout
        assert "summary:" in run.stdout

        make_stub(bins, "ip", 'printf "lo UNKNOWN 10.0.0.1/8\\n"\n')
        make_stub(bins, "ethtool", "exit 1\n")
        run = run_preflight(bins, envfile)
        assert run.returncode == 2
        assert "FAIL  ethtool failed for lo" in run.stdout
        assert "summary:" in run.stdout

        sentinel = root / "sentinel.txt"
        sentinel.write_text("DO_NOT_MODIFY\n", encoding="utf-8")
        make_stub(bins, "ethtool", 'printf "Speed: 200000Mb/s\\nLink detected: yes\\n"\n')
        env = dict(os.environ)
        env["PATH"] = f"{bins}:/usr/bin:/bin"
        launcher = (
            'ln -s -- "$1" "/tmp/ninfer_glm53_ip.$$" || exit 77; '
            'exec /bin/bash "$2" head "$3"'
        )
        proc = subprocess.Popen(
            ["/bin/bash", "-c", launcher, "probe", str(sentinel), str(SCRIPT), str(envfile)],
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        temp_link = pathlib.Path(f"/tmp/ninfer_glm53_ip.{proc.pid}")
        try:
            proc.communicate(timeout=10)
            assert sentinel.read_text(encoding="utf-8") == "DO_NOT_MODIFY\n"
        finally:
            temp_link.unlink(missing_ok=True)

    print("preflight control-flow tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
