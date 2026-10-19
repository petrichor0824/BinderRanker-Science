"""Machine-readable provenance for local scientific runs."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from binderranker import __version__
from binderranker.config import FROZEN_SOURCE_SHA256, GeometryConfig
from binderranker.exceptions import OutputExistsError


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_identity() -> dict[str, Any]:
    return {
        "binderranker_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "frozen_source_sha256": dict(FROZEN_SOURCE_SHA256),
    }


def build_provenance(
    *,
    command: str,
    inputs: list[Path],
    config: GeometryConfig,
    binder_chain: str,
    strict: bool,
    target_validation: dict[str, Any],
    valid_count: int,
    error_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        **runtime_identity(),
        "scientific_profile": config.profile,
        "geometry_config": config.as_dict(),
        "binder_chain": binder_chain,
        "strict_validation": strict,
        "input_count": len(inputs),
        "valid_count": valid_count,
        "error_count": error_count,
        "inputs": [
            {
                "name": path.name,
                "resolved_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in inputs
        ],
        "target_pool_validation": target_validation,
    }


def write_json_atomic(path: Path, value: Any, *, overwrite: bool = False) -> None:
    output = Path(path)
    if output.exists() and not overwrite:
        raise OutputExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

