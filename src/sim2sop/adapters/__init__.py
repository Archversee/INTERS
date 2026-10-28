from __future__ import annotations

from sim2sop.adapters.base import Adapter
from sim2sop.adapters.carla import CarlaAdapter
from sim2sop.adapters.csv_generic import GenericCSVAdapter
from sim2sop.adapters.msfs import MSFSAdapter
from sim2sop.adapters.nmea import NMEAAdapter

ADAPTERS: dict[str, type[Adapter]] = {
    a.name: a for a in (MSFSAdapter, CarlaAdapter, NMEAAdapter, GenericCSVAdapter)
}


def get_adapter(name: str) -> Adapter:
    try:
        return ADAPTERS[name]()
    except KeyError:
        raise ValueError(f"Unknown adapter {name!r}; choose from {sorted(ADAPTERS)}") from None
