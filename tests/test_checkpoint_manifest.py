#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "checkpoint_manifest.py"
spec = importlib.util.spec_from_file_location("checkpoint_manifest", SCRIPT)
assert spec and spec.loader
manifest = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = manifest
spec.loader.exec_module(manifest)


def base_config(quant_method: str = manifest.PRODUCT_QUANT_METHOD) -> dict:
    config: dict = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {},
        "quantization_config": {
            "quant_method": quant_method,
            "config_groups": {"group_0": {"format": manifest.PRODUCT_WEIGHT_FORMAT}},
        },
    }
    for dotted, value in manifest.EXPECTED.items():
        if dotted == "architectures.0":
            continue
        node = config
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    config["text_config"]["layer_types"] = [
        "deepseek_sparse_attention" if i >= 3 and (i - 3) % 4 == 0 else "linear_attention"
        for i in range(45)
    ]
    config["text_config"]["mlp_layer_types"] = ["dense" if i < 3 else "sparse" for i in range(45)]
    return config


def write_safetensors(path: pathlib.Path, header: dict, payload: bytes = b"") -> None:
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-(8 + len(raw)) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def run_fixture(root: pathlib.Path) -> tuple[int, dict]:
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(root)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not run.stdout:
        raise AssertionError(f"manifest produced no JSON: exit={run.returncode} stderr={run.stderr}")
    return run.returncode, json.loads(run.stdout)


def write_common(root: pathlib.Path, config: dict) -> None:
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"dummy": "model.safetensors"}}), encoding="utf-8"
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ninfer-manifest-tests-") as td:
        base = pathlib.Path(td)

        modelopt = base / "modelopt"
        modelopt.mkdir()
        write_common(modelopt, base_config("modelopt"))
        write_safetensors(
            modelopt / "model.safetensors",
            {"dummy": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
            b"\x00\x00",
        )
        code, report = run_fixture(modelopt)
        assert code == 2 and not report["valid_checkpoint_contract"]
        assert any("quantization_config.quant_method" in error for error in report["errors"])

        empty = base / "empty"
        empty.mkdir()
        write_common(empty, base_config())
        write_safetensors(empty / "model.safetensors", {})
        code, report = run_fixture(empty)
        assert code == 2 and not report["valid_checkpoint_contract"]
        assert report["index_missing_from_headers"] == ["dummy"]
        assert report["tensor_count"] == 0

        truncated = base / "truncated"
        truncated.mkdir()
        write_common(truncated, base_config())
        write_safetensors(
            truncated / "model.safetensors",
            {"dummy": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
        )
        code, report = run_fixture(truncated)
        assert code == 2 and not report["valid_checkpoint_contract"]
        assert any("data_offsets exceed shard payload" in error for error in report["errors"])

    print("checkpoint manifest negative admission tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
