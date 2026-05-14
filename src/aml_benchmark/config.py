"""Central configuration loader for aml_benchmark.

Resolves all project paths relative to the repository root so that
modules can be called from any working directory.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# Project root is two levels above this file:
#   src/aml_benchmark/config.py  ->  parents[0]=aml_benchmark, [1]=src, [2]=root
_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_CONFIG_DIR: Path = _PROJECT_ROOT / "configs"


def get_project_root() -> Path:
    """Return the absolute path to the repository root."""
    return _PROJECT_ROOT


def load_yaml(config_name: str) -> dict:
    """Load a YAML config file by name (without .yaml extension).

    Parameters
    ----------
    config_name:
        Filename stem, e.g. ``"paths"`` loads ``configs/paths.yaml``.
    """
    path = _CONFIG_DIR / f"{config_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class PathConfig:
    """Resolved absolute paths constructed from ``configs/paths.yaml``.

    Parameters
    ----------
    config_path:
        Optional path to an alternative ``paths.yaml`` file.  When provided,
        that file is loaded directly instead of the default
        ``configs/paths.yaml`` relative to the project root.  This allows
        the same codebase to run in different environments (e.g. local vs.
        Google Colab) by passing a custom config at runtime:

            PathConfig("/content/drive/MyDrive/aml/configs/paths_colab.yaml")

        When ``None`` (default), the standard ``configs/paths.yaml`` is used.

    Path resolution (_resolve)
    --------------------------
    Values in ``paths.yaml`` can be either relative or absolute:

    * **Relative** paths (e.g. ``data/raw``) are resolved against the
      project root (two directories above this file).
    * **Absolute** paths (e.g. ``/content/drive/MyDrive/aml/data/raw``)
      are used as-is, enabling full Colab / remote-drive support.
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            p = Path(config_path)
            if not p.exists():
                raise FileNotFoundError(f"Config file not found: {p}")
            with p.open(encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh)
        else:
            cfg = load_yaml("paths")

        root = get_project_root()

        def _resolve(value: str) -> Path:
            """Return absolute path; relative values are anchored to root."""
            resolved = Path(value)
            return resolved if resolved.is_absolute() else root / resolved

        # Directories
        self.raw_dir: Path = _resolve(cfg["raw_dir"])
        self.processed_dir: Path = _resolve(cfg["processed_dir"])
        self.splits_dir: Path = _resolve(cfg["splits_dir"])
        self.outputs_dir: Path = _resolve(cfg["outputs_dir"])
        self.leaderboard_dir: Path = _resolve(cfg["leaderboard_dir"])

        # Raw input files
        self.transactions_path: Path = self.raw_dir / cfg["transactions_filename"]
        self.accounts_path: Path = self.raw_dir / cfg["accounts_filename"]
        self.patterns_path: Path = self.raw_dir / cfg["patterns_filename"]

        # Processed files
        self.output_transactions_labeled: Path = (
            self.processed_dir / cfg["output_transactions_labeled"]
        )

        # Split files
        self.train_split: Path = self.splits_dir / "train.parquet"
        self.val_split: Path = self.splits_dir / "val.parquet"
        self.test_split: Path = self.splits_dir / "test.parquet"
        self.split_manifest: Path = self.splits_dir / cfg["split_manifest"]

        # Leaderboard
        self.part_a_summary: Path = self.leaderboard_dir / cfg["part_a_summary"]

        # Optional explicit baseline model path (Part B PAI-HNU only).
        # Empty string or missing key -> auto-discovery in callers.
        baseline_raw = cfg.get("baseline_model_path", "") or ""
        self.baseline_model_path: Path | None = (
            _resolve(baseline_raw) if baseline_raw else None
        )

    def validate(self) -> None:
        """Raise FileNotFoundError for any missing raw input."""
        for attr, path in [
            ("transactions_path", self.transactions_path),
            ("accounts_path", self.accounts_path),
            ("patterns_path", self.patterns_path),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Raw input missing - {attr}: {path}\n"
                    "Make sure raw files are in data/raw/."
                )

    def validate_splits(self) -> None:
        """Raise FileNotFoundError if any split file is missing."""
        for attr, path in [
            ("train_split", self.train_split),
            ("val_split", self.val_split),
            ("test_split", self.test_split),
        ]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Split file missing - {attr}: {path}\n"
                    "Run: python -m aml_benchmark.data.splitter"
                )
