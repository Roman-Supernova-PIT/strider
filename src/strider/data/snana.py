"""Read Sundial SNANA FITS pairs without changing their native binning."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from astropy.io import fits

from .records import SourcePair


BLOCK_PATTERN = re.compile(r"_([A-Za-z0-9]+)-(\d{4})_HEAD\.FITS\.gz$")


def discover_source_pairs(source_dir: str | Path) -> list[SourcePair]:
    source = Path(source_dir).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Sundial source directory does not exist: {source}")
    pairs: list[SourcePair] = []
    for head in sorted(source.glob("*_HEAD.FITS.gz")):
        match = BLOCK_PATTERN.search(head.name)
        if match is None:
            continue
        spec = head.with_name(head.name.replace("_HEAD.FITS.gz", "_SPEC.FITS.gz"))
        if not spec.is_file():
            raise FileNotFoundError(f"Missing SPEC partner for {head.name}")
        pairs.append(
            SourcePair(
                block=int(match.group(2)),
                model=match.group(1),
                head_path=str(head),
                spec_path=str(spec),
            )
        )
    if not pairs:
        raise FileNotFoundError(f"No HEAD/SPEC FITS pairs found in {source}")
    return pairs


def read_head_table(pair: SourcePair) -> pd.DataFrame:
    wanted = [
        "SNID",
        "SIM_TYPE_INDEX",
        "SIM_TEMPLATE_INDEX",
        "SIM_REDSHIFT_CMB",
        "PEAKMJD",
        "SIM_PEAKMJD",
    ]
    with fits.open(pair.head_path, mode="readonly", memmap=False) as hdus:
        table = hdus[1].data
        missing = [name for name in wanted if name not in table.names]
        if missing:
            raise ValueError(f"{pair.head_path} lacks columns: {missing}")
        # FITS tables are big-endian. Explicit native dtypes avoid passing
        # byte-swapped arrays into pandas and Parquet.
        frame = pd.DataFrame(
            {
                "SNID": np.asarray(table["SNID"], dtype=np.int64),
                "SIM_TYPE_INDEX": np.asarray(table["SIM_TYPE_INDEX"], dtype=np.int64),
                "SIM_TEMPLATE_INDEX": np.asarray(
                    table["SIM_TEMPLATE_INDEX"], dtype=np.int64
                ),
                "SIM_REDSHIFT_CMB": np.asarray(
                    table["SIM_REDSHIFT_CMB"], dtype=np.float64
                ),
                "PEAKMJD": np.asarray(table["PEAKMJD"], dtype=np.float64),
                "SIM_PEAKMJD": np.asarray(table["SIM_PEAKMJD"], dtype=np.float64),
            }
        )
    frame = frame.rename(
        columns={
            "SNID": "snid",
            "SIM_TYPE_INDEX": "gentype",
            "SIM_TEMPLATE_INDEX": "template_index",
            "SIM_REDSHIFT_CMB": "redshift",
            "PEAKMJD": "estimated_peak_mjd",
            "SIM_PEAKMJD": "simulation_peak_mjd",
        }
    )
    frame["snid"] = frame["snid"].astype(np.int64)
    frame["block"] = pair.block
    frame["model"] = pair.model
    frame["head_path"] = pair.head_path
    frame["spec_path"] = pair.spec_path
    return frame


def inspect_pairs(pairs: Iterable[SourcePair]) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        head = read_head_table(pair)
        rows.append(
            {
                "block": pair.block,
                "model": pair.model,
                "objects": len(head),
                "redshift_median": float(head["redshift"].median()),
                "redshift_at_least_2": int((head["redshift"] >= 2.0).sum()),
                "head_path": pair.head_path,
            }
        )
    return pd.DataFrame(rows).sort_values(["block", "model"]).reset_index(drop=True)


def read_spectrum_snids(pair: SourcePair) -> set[int]:
    """Return catalogue identifiers that have at least one stored spectrum."""
    with fits.open(pair.spec_path, mode="readonly", memmap=False) as hdus:
        return set(np.asarray(hdus[1].data["SNID"], dtype=np.int64).tolist())


def read_selected_spectra(
    pair: SourcePair,
    selected_snids: set[int],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Return selected observations and concatenated native-bin arrays.

    SNANA spectrum pointers are one-based and inclusive.  Converting the lower
    pointer to zero-based and using the upper pointer as Python's exclusive end
    keeps exactly NBIN_LAM rows.
    """
    observations: list[dict] = []
    wavelength_min: list[np.ndarray] = []
    wavelength_max: list[np.ndarray] = []
    observed_flux: list[np.ndarray] = []
    flux_error: list[np.ndarray] = []
    clean_flux: list[np.ndarray] = []

    next_bin = 0
    with fits.open(pair.spec_path, mode="readonly", memmap=False) as hdus:
        summary = hdus[1].data
        bins = hdus[2].data
        for row in summary:
            snid = int(row["SNID"])
            if snid not in selected_snids:
                continue
            start = int(row["PTRSPEC_MIN"]) - 1
            stop = int(row["PTRSPEC_MAX"])
            expected = int(row["NBIN_LAM"])
            if start < 0 or stop <= start or stop > len(bins):
                raise ValueError(f"Invalid spectrum pointers for SNID {snid}: {start + 1}:{stop}")
            native = bins[start:stop]
            if len(native) != expected:
                raise ValueError(
                    f"SNID {snid} says NBIN_LAM={expected}, pointers select {len(native)} rows"
                )
            first_bin = next_bin
            wavelength_min.append(np.asarray(native["LAMMIN"], dtype=np.float32))
            wavelength_max.append(np.asarray(native["LAMMAX"], dtype=np.float32))
            observed_flux.append(np.asarray(native["FLAM"], dtype=np.float32))
            flux_error.append(np.asarray(native["FLAMERR"], dtype=np.float32))
            clean_flux.append(np.asarray(native["SIM_FLAM"], dtype=np.float32))
            observations.append(
                {
                    "snid": snid,
                    "mjd": float(row["MJD"]),
                    "exposure_seconds": float(row["Texpose"]),
                    "first_bin": first_bin,
                    "bin_count": expected,
                }
            )
            next_bin += expected

    arrays = {
        "wavelength_min": _concatenate(wavelength_min),
        "wavelength_max": _concatenate(wavelength_max),
        "observed_flux": _concatenate(observed_flux),
        "flux_error": _concatenate(flux_error),
        "clean_flux": _concatenate(clean_flux),
    }
    return observations, arrays


def _concatenate(parts: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.empty(0, np.float32)
