#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    stub = r'''
#include "ninfer_glm53/checkpoint_binding.hpp"

namespace ninfer::glm53 {

std::string_view to_string(CheckpointKind) noexcept { return "TEST_DOUBLE"; }

BindingReport bind_checkpoint_directory(const std::filesystem::path&, bool) {
    BindingReport report;
    report.complete = true;
    return report;
}

std::string binding_receipt_json(const BindingReport&) {
    return "{\"test_double\":true,\"complete\":true}\n";
}

}  // namespace ninfer::glm53
'''
    with tempfile.TemporaryDirectory(prefix="ninfer-bind-cli-") as td:
        root = pathlib.Path(td)
        stub_path = root / "stub.cpp"
        stub_path.write_text(stub, encoding="utf-8")
        exe = root / "ninfer-bind-cli-test"
        subprocess.run(
            [
                "c++",
                "-std=c++20",
                "-O2",
                "-DNDEBUG",
                "-I",
                str(ROOT / "include"),
                str(ROOT / "apps" / "ninfer_glm53_bind.cpp"),
                str(stub_path),
                "-o",
                str(exe),
            ],
            check=True,
            cwd=ROOT,
        )

        receipt = root / "receipt.json"
        run = subprocess.run(
            [str(exe), "TEST_DOUBLE", "-o", str(receipt)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 0
        assert '"complete":true' in receipt.read_text(encoding="utf-8")

        run = subprocess.run(
            [str(exe), "TEST_DOUBLE", "-o", "/dev/full"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert run.returncode == 1
        assert "failed while writing /dev/full" in run.stderr

        with open("/dev/full", "w", encoding="utf-8") as sink:
            run = subprocess.run(
                [str(exe), "TEST_DOUBLE"],
                stdout=sink,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
        assert run.returncode == 1
        assert "failed while writing receipt to stdout" in run.stderr

    print("binding CLI output failure tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
