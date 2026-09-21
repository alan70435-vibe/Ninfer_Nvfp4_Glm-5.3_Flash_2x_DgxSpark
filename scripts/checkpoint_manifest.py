#!/usr/bin/env python3
"""Inventory and validate a local GLM-5.3-Flash checkpoint without loading tensor payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

EXPECTED = {
    "architectures.0": "Glm5NextForConditionalGeneration",
    "text_config.hidden_size": 4096,
    "text_config.intermediate_size": 12288,
    "text_config.num_hidden_layers": 45,
    "text_config.num_attention_heads": 64,
    "text_config.num_key_value_heads": 64,
    "text_config.max_position_embeddings": 1048576,
    "text_config.vocab_size": 154880,
    "text_config.q_lora_rank": 1536,
    "text_config.kv_lora_rank": 512,
    "text_config.qk_head_dim": 256,
    "text_config.qk_nope_head_dim": 256,
    "text_config.qk_rope_head_dim": 0,
    "text_config.v_head_dim": 256,
    "text_config.index_n_heads": 32,
    "text_config.index_head_dim": 128,
    "text_config.index_topk": 2048,
    "text_config.index_kpool": 4,
    "text_config.n_routed_experts": 288,
    "text_config.n_shared_experts": 1,
    "text_config.num_experts_per_tok": 8,
    "text_config.moe_intermediate_size": 2048,
    "text_config.first_k_dense_replace": 3,
    "text_config.mhc": True,
    "text_config.hc_mult": 4,
    "text_config.hc_eps": 1e-6,
    "text_config.hc_sinkhorn_iters": 20,
    "text_config.swiglu_limit": 10.0,
    "text_config.routed_scaling_factor": 2.5,
    "text_config.moe_router_dtype": "float32",
    "text_config.scoring_func": "sigmoid",
    "text_config.topk_method": "noaux_tc",
    "text_config.mla_use_nope": True,
    "text_config.index_kpool_compress": True,
    "text_config.index_kpool_always_select_tail": True,
    "text_config.index_share_for_mtp_iteration": True,
    "text_config.indexer_rope_interleave": True,
    "text_config.linear_attn_config.num_heads": 64,
    "text_config.linear_attn_config.head_dim": 128,
    "text_config.linear_attn_config.short_conv_kernel_size": 4,
    "text_config.linear_attn_config.gate_lower_bound": -5.0,
}

INDEX_CANDIDATES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)

MAX_SAFETENSORS_HEADER_BYTES = 256 * 1024 * 1024


def dig(data: Any, dotted: str) -> Any:
    cur = data
    for part in dotted.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    """Read only the JSON header of a safetensors file."""
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("file is shorter than safetensors framing")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len == 0 or header_len > MAX_SAFETENSORS_HEADER_BYTES:
            raise ValueError(f"invalid safetensors header length {header_len}")
        raw_header = f.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError("truncated safetensors JSON header")
    header = json.loads(raw_header)
    if not isinstance(header, dict):
        raise ValueError("safetensors header must be a JSON object")
    return header


@dataclass(frozen=True)
class FileRecord:
    name: str
    size: int
    sha256: str | None
    tensor_count: int | None
    header_error: str | None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="local checkpoint directory")
    parser.add_argument("--hash-shards", action="store_true", help="SHA-256 all weight shards (can be slow)")
    parser.add_argument("--output", type=Path, help="write JSON report here")
    args = parser.parse_args()

    root = args.checkpoint.expanduser().resolve()
    config_path = root / "config.json"
    if not config_path.is_file():
        parser.error(f"missing {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    observed: dict[str, Any] = {}
    for key, expected in EXPECTED.items():
        actual = dig(config, key)
        observed[key] = actual
        if actual != expected:
            errors.append(f"{key}: expected {expected!r}, got {actual!r}")

    expected_layer_types = [
        "deepseek_sparse_attention" if i >= 3 and (i - 3) % 4 == 0 else "linear_attention"
        for i in range(45)
    ]
    expected_mlp_types = ["dense" if i < 3 else "sparse" for i in range(45)]
    actual_layer_types = dig(config, "text_config.layer_types")
    actual_mlp_types = dig(config, "text_config.mlp_layer_types")
    if actual_layer_types != expected_layer_types:
        errors.append("text_config.layer_types does not match the 34 KDA / 11 sparse-MLA topology")
    if actual_mlp_types != expected_mlp_types:
        errors.append("text_config.mlp_layer_types does not match the 3 dense / 42 sparse-MoE topology")

    index_path = next((root / name for name in INDEX_CANDIDATES if (root / name).is_file()), None)
    weight_map: dict[str, str] = {}
    if index_path:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        raw_map = index.get("weight_map", {})
        if not isinstance(raw_map, dict):
            errors.append(f"{index_path.name}: weight_map is not an object")
        else:
            weight_map = {str(k): str(v) for k, v in raw_map.items()}

    referenced_shards = sorted(set(weight_map.values()))
    if not referenced_shards:
        referenced_shards = sorted(p.name for p in root.glob("*.safetensors"))
        if not referenced_shards:
            referenced_shards = sorted(p.name for p in root.glob("*.bin"))

    if not referenced_shards:
        errors.append("no weight shards or weight index found")

    files: list[FileRecord] = []
    missing_shards: list[str] = []
    actual_tensor_records: dict[str, dict[str, Any]] = {}
    tensor_owner_from_headers: dict[str, str] = {}
    duplicate_header_tensors: list[str] = []
    shard_header_errors: list[str] = []

    for name in referenced_shards:
        path = root / name
        if not path.is_file():
            missing_shards.append(name)
            continue

        header_tensor_count: int | None = None
        header_error: str | None = None
        if path.suffix == ".safetensors":
            try:
                header = read_safetensors_header(path)
                tensors = {k: v for k, v in header.items() if k != "__metadata__"}
                header_tensor_count = len(tensors)
                for tensor_name, metadata in tensors.items():
                    if tensor_name in tensor_owner_from_headers:
                        duplicate_header_tensors.append(tensor_name)
                        continue
                    tensor_owner_from_headers[tensor_name] = name
                    if isinstance(metadata, dict):
                        actual_tensor_records[tensor_name] = {
                            "dtype": metadata.get("dtype"),
                            "shape": metadata.get("shape"),
                        }
                    else:
                        actual_tensor_records[tensor_name] = {"invalid_metadata": True}
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                header_error = str(exc)
                shard_header_errors.append(f"{name}: {exc}")

        files.append(
            FileRecord(
                name=name,
                size=path.stat().st_size,
                sha256=sha256(path) if args.hash_shards else None,
                tensor_count=header_tensor_count,
                header_error=header_error,
            )
        )

    if missing_shards:
        errors.append(f"missing {len(missing_shards)} referenced weight shard(s)")
    if duplicate_header_tensors:
        errors.append(f"{len(duplicate_header_tensors)} tensor name(s) occur in multiple shard headers")
    if shard_header_errors:
        errors.extend(f"safetensors header error: {item}" for item in shard_header_errors)

    index_missing_from_headers: list[str] = []
    index_owner_mismatches: list[dict[str, str]] = []
    header_unindexed: list[str] = []
    if weight_map and actual_tensor_records and not shard_header_errors:
        for tensor_name, expected_owner in weight_map.items():
            actual_owner = tensor_owner_from_headers.get(tensor_name)
            if actual_owner is None:
                index_missing_from_headers.append(tensor_name)
            elif actual_owner != expected_owner:
                index_owner_mismatches.append(
                    {"tensor": tensor_name, "index_owner": expected_owner, "header_owner": actual_owner}
                )
        header_unindexed = sorted(set(actual_tensor_records) - set(weight_map))
        if index_missing_from_headers:
            errors.append(f"{len(index_missing_from_headers)} indexed tensor(s) are absent from shard headers")
        if index_owner_mismatches:
            errors.append(f"{len(index_owner_mismatches)} indexed tensor(s) point at the wrong shard")
        if header_unindexed:
            warnings.append(f"{len(header_unindexed)} header tensor(s) are not present in the weight_map")

    tensor_names = sorted(actual_tensor_records or {name: {} for name in weight_map})
    prefixes: dict[str, int] = {}
    for name in tensor_names:
        prefix = name.split(".", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1

    released_markers = (".hc_attn_fn", ".hc_ffn_fn", ".self_attn.A_log")
    transformers_markers = (".attn_hc.fn", ".ffn_hc.fn", ".self_attn.forget_gate.A_log")
    released_hits = sum(any(marker in name for marker in released_markers) for name in tensor_names)
    transformers_hits = sum(any(marker in name for marker in transformers_markers) for name in tensor_names)
    if released_hits and not transformers_hits:
        namespace_profile = "released_glm5_next"
    elif transformers_hits and not released_hits:
        namespace_profile = "transformers_saved"
    elif released_hits and transformers_hits:
        namespace_profile = "mixed_glm5_next_names"
        warnings.append("checkpoint contains both released and Transformers-saved GLM naming markers")
    elif tensor_names:
        namespace_profile = "custom_or_converted"
    else:
        namespace_profile = "unindexed"

    dtype_counts = Counter(
        str(record.get("dtype")) for record in actual_tensor_records.values() if record.get("dtype") is not None
    )
    schema_rows = [
        {"name": name, "dtype": actual_tensor_records[name].get("dtype"), "shape": actual_tensor_records[name].get("shape")}
        for name in sorted(actual_tensor_records)
    ]
    schema_bytes = json.dumps(schema_rows, separators=(",", ":"), sort_keys=True).encode("utf-8")
    tensor_schema_sha256 = hashlib.sha256(schema_bytes).hexdigest() if schema_rows else None

    report = {
        "schema": 2,
        "checkpoint": str(root),
        "valid_checkpoint_contract": not errors,
        "errors": errors,
        "warnings": warnings,
        "config_sha256": sha256(config_path),
        "index_file": index_path.name if index_path else None,
        "index_sha256": sha256(index_path) if index_path else None,
        "expected_fields": EXPECTED,
        "observed_fields": observed,
        "tensor_count": len(tensor_names),
        "tensor_schema_sha256": tensor_schema_sha256,
        "tensor_dtype_counts": dict(sorted(dtype_counts.items())),
        "tensor_namespace_profile": namespace_profile,
        "tensor_top_level_prefix_counts": prefixes,
        "weight_shards": [asdict(record) for record in files],
        "weight_bytes": sum(record.size for record in files),
        "missing_shards": missing_shards,
        "index_missing_from_headers": index_missing_from_headers,
        "index_owner_mismatches": index_owner_mismatches,
        "header_unindexed": header_unindexed,
    }

    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
