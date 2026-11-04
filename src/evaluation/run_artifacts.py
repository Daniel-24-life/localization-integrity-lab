"""Small, append-only local experiment record. No external services."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from uuid import uuid4
import contextlib
import hashlib
import json
import platform
import sys
import traceback
import shutil
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def load_config():
    return yaml.safe_load((ROOT / "configs/default_config.yaml").read_text(encoding="utf-8"))


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()


class RunSession:
    def __init__(self, kind, config=None, inputs=()):
        run_id = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S") + "_" + kind + "_" + uuid4().hex[:8]
        self.path = ROOT / "results/runs" / run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = load_config() if config is None else config
        config_text = yaml.safe_dump(self.config, allow_unicode=True, sort_keys=True)
        (self.path / "config.yaml").write_text(config_text, encoding="utf-8")
        source_files = [*ROOT.glob("*.py"), *ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"), *ROOT.glob("configs/*.yaml"), *ROOT.glob("requirements.txt"), *ROOT.glob("pytest.ini")]
        sources = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(source_files)}
        write_json(self.path / "source_hashes.json", sources)
        for p in source_files:
            target = self.path / "source" / p.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, target)
        environment = {}
        for name in ["numpy", "scipy", "pandas", "matplotlib", "PyYAML", "rosbags", "pytest", "tabulate", "tzdata"]:
            try:
                environment[name] = version(name)
            except PackageNotFoundError:
                environment[name] = None  # Optional tools need not block unrelated experiments.
        write_json(self.path / "environment.json", environment)
        self.record = {"run_id": run_id, "kind": kind, "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                       "status": "running", "command": sys.argv, "python": sys.version, "platform": platform.platform(),
                       "numpy": np.__version__, "seed": self.config.get("project", {}).get("random_seed"),
                       "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(), "inputs": []}
        for path in inputs:
            p = Path(path).resolve()
            self.record["inputs"].append({"path": str(p), "sha256": digest(p), "bytes": p.stat().st_size})
            sidecar = p.with_suffix(".metadata.json")
            if sidecar.exists():
                self.record["inputs"].append({"path": str(sidecar), "sha256": digest(sidecar), "bytes": sidecar.stat().st_size})
        write_json(self.path / "manifest.json", self.record)

    def __enter__(self):
        self.log = (self.path / "run.log").open("w", encoding="utf-8")
        self.stack = contextlib.ExitStack()
        self.stack.enter_context(contextlib.redirect_stdout(_Tee(sys.stdout, self.log)))
        self.stack.enter_context(contextlib.redirect_stderr(_Tee(sys.stderr, self.log)))
        print("Run directory:", self.path)
        return self

    def save_sequence(self, tag, data, result, metrics):
        arrays = {key: value for key, value in vars(data).items() if isinstance(value, np.ndarray)}
        arrays.update({"estimate_" + key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        arrays.update({"evaluation_" + key: value for key, value in metrics.items() if isinstance(value, np.ndarray)})
        np.savez_compressed(self.path / f"{tag}_frames.npz", **arrays)
        write_json(self.path / f"{tag}_metrics.json", {key: value for key, value in metrics.items() if not isinstance(value, np.ndarray)})
        write_json(self.path / f"{tag}_data_metadata.json", data.metadata)

    def __exit__(self, exc_type, exc, tb):
        if exc:
            traceback.print_exception(exc_type, exc, tb)
        self.stack.close()
        self.log.close()
        self.record["status"] = "failed" if exc else "completed"
        self.record["completed_at"] = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
        self.record["outputs"] = {p.relative_to(self.path).as_posix(): digest(p) for p in sorted(self.path.rglob("*")) if p.is_file() and p.name != "manifest.json"}
        write_json(self.path / "manifest.json", self.record)
