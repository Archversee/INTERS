"""Synthetic run generators. Each writes files in the real simulator's own format so
recorded runs can replace them without code changes."""

from __future__ import annotations

from pathlib import Path

from sim2sop import config


def generate(domain: str, n: int = 5, seed: int | None = None, out_dir: Path | None = None,
             scenario: str | None = None) -> list[Path]:
    """`scenario="obstacle"` generates the post-NOTAM runs (air only). seed=None uses
    each generator's default, so baseline and obstacle runs differ."""
    out_dir = out_dir or config.RAW_DIR / domain
    if domain == "air":
        from sim2sop.generate.air import generate_air_runs, generate_obstacle_runs
        kw = {"seed": seed} if seed is not None else {}
        if scenario == "obstacle":
            return generate_obstacle_runs(out_dir, n=n, **kw)
        return generate_air_runs(out_dir, n=n, **kw)
    # TODO(data): land (CARLA vehicle state) and sea (NMEA 0183) generators.
    raise NotImplementedError(f"No generator for domain {domain!r} yet (air is implemented)")
