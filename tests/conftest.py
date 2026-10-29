from pathlib import Path

import pytest

from sim2sop import config
from sim2sop.generate.air import generate_air_runs


@pytest.fixture
def data_dir(tmp_path, monkeypatch) -> Path:
    """Point all storage at a temp folder."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "GUIDELINES_DIR", tmp_path / "guidelines")
    monkeypatch.setattr(config, "LIBRARY_DIR", tmp_path / "references")
    monkeypatch.setattr(config, "MEMORY_DIR", tmp_path / "memory")
    return tmp_path


@pytest.fixture(scope="session")
def air_raw(tmp_path_factory) -> list[Path]:
    """Five generated C172 circuits (session-wide: generation is the slow part)."""
    return generate_air_runs(tmp_path_factory.mktemp("raw") / "air", n=5)


@pytest.fixture
def refs_dir() -> Path:
    return config.REFERENCES_DIR / "air"
