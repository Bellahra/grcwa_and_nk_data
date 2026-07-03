#!/usr/bin/env python3
"""
Download n,k data from refractiveindex.info and resample it on a wavelength grid.

Example:
    python3 fetch_nk_refractiveindex.py --shelf main --book Si --page Franta-250K \
        --material "silicon (Si)" --start-nm 30 --stop-nm 200 --step-nm 5 \
        --output n_k_silicon.dat
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
from dataclasses import dataclass
from typing import Iterable


DATABASE_URL = "https://refractiveindex.info/database/data/{shelf}/{book}/nk/{page}.yml"
PAGE_URL = "https://refractiveindex.info/?shelf={shelf}&book={book}&page={page}"


@dataclass(frozen=True)
class NkRecord:
    source_url: str
    temperature_k: str | None
    rows_um_n_k: list[tuple[float, float, float]]


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "nk-fetcher/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8")


def extract_temperature_k(text: str) -> str | None:
    match = re.search(r"^\s*temperature:\s*([0-9.+-]+)\s*$", text, re.M)
    return match.group(1) if match else None


def parse_nk_record(text: str, source_url: str) -> NkRecord:
    data_match = re.search(
        r"^\s*-\s*type:\s*tabulated nk\s*\n\s*data:\s*\|\n"
        r"(?P<data>(?:^\s+[0-9.+\-Ee]+\s+[0-9.+\-Ee]+\s+[0-9.+\-Ee]+\s*$\n?)*)",
        text,
        re.M,
    )
    if not data_match:
        raise ValueError("Could not find a 'tabulated nk' data block in the record.")

    rows: list[tuple[float, float, float]] = []
    for line in data_match.group("data").splitlines():
        parts = line.split()
        if len(parts) == 3:
            wavelength_um, n_value, k_value = map(float, parts)
            rows.append((wavelength_um, n_value, k_value))

    if len(rows) < 2:
        raise ValueError("The nk data block did not contain enough rows to interpolate.")

    rows.sort(key=lambda row: row[0])
    return NkRecord(
        source_url=source_url,
        temperature_k=extract_temperature_k(text),
        rows_um_n_k=rows,
    )


def wavelength_grid_nm(start_nm: float, stop_nm: float, step_nm: float) -> list[float]:
    if step_nm <= 0:
        raise ValueError("--step-nm must be positive.")
    values: list[float] = []
    current = start_nm
    epsilon = abs(step_nm) * 1e-9
    while current <= stop_nm + epsilon:
        values.append(round(current, 12))
        current += step_nm
    return values


def interpolate_or_extrapolate(x: float, points: list[tuple[float, float]]) -> float:
    if x <= points[0][0]:
        x0, y0 = points[0]
        x1, y1 = points[1]
    elif x >= points[-1][0]:
        x0, y0 = points[-2]
        x1, y1 = points[-1]
    else:
        lo = 0
        hi = len(points) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if points[mid][0] <= x:
                lo = mid
            else:
                hi = mid
        x0, y0 = points[lo]
        x1, y1 = points[hi]

    if x1 == x0:
        return y0
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def resample(rows_um_n_k: Iterable[tuple[float, float, float]], grid_nm: Iterable[float]) -> list[tuple[float, float, float]]:
    rows_nm_n_k = [(um * 1000.0, n, k) for um, n, k in rows_um_n_k]
    n_points = [(wavelength_nm, n) for wavelength_nm, n, _ in rows_nm_n_k]
    k_points = [(wavelength_nm, k) for wavelength_nm, _, k in rows_nm_n_k]
    return [
        (
            wavelength_nm,
            interpolate_or_extrapolate(wavelength_nm, n_points),
            interpolate_or_extrapolate(wavelength_nm, k_points),
        )
        for wavelength_nm in grid_nm
    ]


def format_nm(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.12g}"


def write_dat(
    output_path: str,
    record: NkRecord,
    sampled_rows: list[tuple[float, float, float]],
    material: str,
    shelf: str,
    book: str,
    page: str,
) -> None:
    source_min_nm = record.rows_um_n_k[0][0] * 1000.0
    source_max_nm = record.rows_um_n_k[-1][0] * 1000.0
    grid_min_nm = sampled_rows[0][0]
    grid_max_nm = sampled_rows[-1][0]
    extrapolated = grid_min_nm < source_min_nm or grid_max_nm > source_max_nm
    temperature = f"{record.temperature_k} K" if record.temperature_k else "unknown"
    page_url = PAGE_URL.format(shelf=shelf, book=book, page=page)

    with open(output_path, "w", encoding="utf-8") as file:
        header = (
            f"# material={material}; shelf={shelf}; book={book}; page={page}; "
            f"temperature={temperature}; wavelength_unit=nm; columns=wavelength_nm n k; "
            f"source_range_nm={source_min_nm:.6g}-{source_max_nm:.6g}; "
            f"extrapolated_endpoint={str(extrapolated).lower()}; source={page_url}\n"
        )
        file.write(header)
        for wavelength_nm, n_value, k_value in sampled_rows:
            file.write(f"{format_nm(wavelength_nm):>8s} {n_value:.12g} {k_value:.12g}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shelf", default="main", help="refractiveindex.info shelf name")
    parser.add_argument("--book", default="Si", help="refractiveindex.info book/material id")
    parser.add_argument("--page", default="Franta-250K", help="refractiveindex.info page id")
    parser.add_argument("--material", default="silicon (Si)", help="human-readable material name")
    parser.add_argument("--start-nm", type=float, default=30.0, help="first wavelength in nm")
    parser.add_argument("--stop-nm", type=float, default=200.0, help="last wavelength in nm")
    parser.add_argument("--step-nm", type=float, default=5.0, help="wavelength step in nm")
    parser.add_argument("--output", default="n_k_silicon.dat", help="output .dat file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_url = DATABASE_URL.format(shelf=args.shelf, book=args.book, page=args.page)

    try:
        record = parse_nk_record(fetch_text(data_url), data_url)
        grid = wavelength_grid_nm(args.start_nm, args.stop_nm, args.step_nm)
        sampled_rows = resample(record.rows_um_n_k, grid)
        write_dat(args.output, record, sampled_rows, args.material, args.shelf, args.book, args.page)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(sampled_rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
