"""Synthetic run generators. Each writes files in the real simulator's own format so
recorded runs can replace them without code changes."""

from __future__ import annotations

from pathlib import Path

from sim2sop import config


def generate(domain: str, n: int = 5, seed: int = 172, out_dir: Path | None = None) -> list[Path]:
    out_dir = out_dir or config.RAW_DIR / domain
    if domain == "air":
        from sim2sop.generate.air import generate_air_runs
        return generate_air_runs(out_dir, n=n, seed=seed)
    # TODO(data): land (CARLA vehicle state) and sea (NMEA 0183) generators.
    raise NotImplementedError(f"No generator for domain {domain!r} yet (air is implemented)")
