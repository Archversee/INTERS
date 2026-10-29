"""Synthetic run generators. Each writes files in the real simulator's own format so
recorded runs can replace them without code changes."""

from __future__ import annotations

from pathlib import Path

from sim2sop import config


def generate(domain: str, n: int = 5, seed: int | None = None, out_dir: Path | None = None) -> list[Path]:
    """seed=None uses the generator's default seed."""
    out_dir = out_dir or config.RAW_DIR / domain
    kw = {"seed": seed} if seed is not None else {}
    if domain == "air":
        from sim2sop.generate.air import generate_air_runs
        return generate_air_runs(out_dir, n=n, **kw)
    if domain == "land":
        from sim2sop.generate.land import generate_land_runs
        return generate_land_runs(out_dir, n=n, **kw)
    # TODO(data): sea (NMEA 0183) generator.
    raise NotImplementedError(f"No generator for domain {domain!r} yet (air and land are implemented)")
