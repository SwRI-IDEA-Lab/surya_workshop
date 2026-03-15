#!/usr/bin/env python3
"""Benchmark boto3 S3 download throughput to find optimal transfer settings
for use with HelioNetCDFDataset.

Measures throughput by downloading byte ranges in parallel — exactly what boto3's
transfer manager does internally — so any (concurrency, part_size) pair can be
tested without downloading the full file each time.

Usage:
    python -m workshop_infrastructure.benchmark_s3 s3://bucket/path/file.nc
    python -m workshop_infrastructure.benchmark_s3 s3://bucket/path/file.nc --anon --quick
    python -m workshop_infrastructure.benchmark_s3 s3://bucket/path/file.nc --output-csv results.csv
"""

import argparse
import csv
import os
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from workshop_infrastructure.utils import detect_ec2_region

try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config as BotoConfig
except ImportError:
    print("ERROR: boto3 is required. Install with: pip install boto3")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Default benchmark grids
# ---------------------------------------------------------------------------

FULL_CONCURRENCIES = [1, 2, 4, 8, 16]
FULL_PART_SIZES_MB = [8, 16, 32, 64, 128]   # stops at 128: larger parts rarely help over internet

QUICK_CONCURRENCIES = [1, 4, 16]
QUICK_PART_SIZES_MB = [8, 32, 128]           # covers small, medium, large

_EPILOG = """
Examples:

  # Default full grid (public bucket):
  python -m workshop_infrastructure.benchmark_s3 \\
      s3://my-bucket/data/sample.nc --anon

  # Quick 6-cell grid, save results to CSV:
  python -m workshop_infrastructure.benchmark_s3 \\
      s3://my-bucket/data/sample.nc --quick --output-csv bench.csv

  # Custom grid for an EC2 instance where high concurrency may help:
  python -m workshop_infrastructure.benchmark_s3 \\
      s3://my-bucket/data/sample.nc \\
      --concurrencies 8 16 32 --part-sizes-mb 64 128 256

  # Faster run with a smaller sample (less stable, but quicker):
  python -m workshop_infrastructure.benchmark_s3 \\
      s3://my-bucket/data/sample.nc --sample-mb 128 --quick
"""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CellResult:
    concurrency: int
    part_size_mb: int
    throughput_mbps: float    # median across repeats; NaN on error
    sample_mb: float          # actual bytes sampled, in MB
    n_parts: int              # number of byte-range requests issued
    single_part: bool         # True when only 1 request was needed
    error: str | None = None  # set when all repeats failed


@dataclass
class BenchmarkConfig:
    s3_uri: str
    sample_mb: int = 512
    concurrencies: list[int] = field(default_factory=lambda: list(FULL_CONCURRENCIES))
    part_sizes_mb: list[int] = field(default_factory=lambda: list(FULL_PART_SIZES_MB))
    n_repeats: int = 3
    anon: bool = False
    region: str | None = None
    output_csv: str | None = None
    warmup: bool = True


# ---------------------------------------------------------------------------
# AWS client helpers
# ---------------------------------------------------------------------------

def _make_client(anon: bool, region: str | None, pool_size: int = 32):
    """Return a boto3 S3 client with the given signature and connection pool."""
    retry_cfg = {"max_attempts": 3, "mode": "adaptive"}
    if anon:
        return boto3.client(
            "s3",
            region_name=region,
            config=BotoConfig(
                signature_version=UNSIGNED,
                max_pool_connections=pool_size,
                retries=retry_cfg,
            ),
        )
    return boto3.client(
        "s3",
        region_name=region,
        config=BotoConfig(max_pool_connections=pool_size, retries=retry_cfg),
    )


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri!r}")
    bucket, key = uri[5:].split("/", 1)
    return bucket, key


def _probe_file_size(client, bucket: str, key: str) -> int:
    """Return the object size in bytes via HeadObject."""
    try:
        resp = client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read s3://{bucket}/{key}: {exc}\n"
            "Check that the URI is correct and that you have s3:GetObject permission "
            "(or pass --anon for public buckets)."
        ) from exc
    return resp["ContentLength"]


# ---------------------------------------------------------------------------
# Core download primitive
# ---------------------------------------------------------------------------

def _fetch_range(client, bucket: str, key: str, start: int, end: int) -> int:
    """Download bytes [start, end) from S3, discard the data, return byte count.

    The response body must be fully consumed or boto3 will leave the underlying
    connection in an unusable state.  We drain in 1 MB chunks to bound memory
    usage: at 32 concurrent threads each holding one chunk, peak overhead is ~32 MB.
    """
    response = client.get_object(Bucket=bucket, Key=key, Range=f"bytes={start}-{end - 1}")
    body = response["Body"]
    received = 0
    for chunk in iter(lambda: body.read(1 * 1024 * 1024), b""):
        received += len(chunk)
    return received


# ---------------------------------------------------------------------------
# Single-cell benchmark
# ---------------------------------------------------------------------------

def _benchmark_cell(
    anon: bool,
    region: str | None,
    bucket: str,
    key: str,
    file_size: int,
    concurrency: int,
    part_size_mb: int,
    sample_bytes: int,
) -> tuple[float, int, bool]:
    """Download sample_bytes using parallel range requests and return (throughput_mbps, n_parts, single_part).

    A fresh boto3 client is created for each cell so that TCP connections from
    a prior measurement do not inflate subsequent results.
    """
    part_size = part_size_mb * 1024 * 1024

    # Build non-overlapping byte ranges covering [0, sample_bytes)
    ranges = []
    offset = 0
    while offset < sample_bytes:
        end = min(offset + part_size, sample_bytes)
        ranges.append((offset, end))
        offset = end

    single_part = len(ranges) == 1
    # No benefit from more threads than parts
    actual_concurrency = min(concurrency, len(ranges))
    pool_size = max(32, actual_concurrency * 2)
    client = _make_client(anon, region, pool_size)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=actual_concurrency) as executor:
        futures = [
            executor.submit(_fetch_range, client, bucket, key, s, e)
            for s, e in ranges
        ]
        total_bytes = sum(f.result() for f in as_completed(futures))
    elapsed = time.perf_counter() - t0

    throughput = total_bytes / elapsed / 1e6  # MB/s
    return throughput, len(ranges), single_part


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------

def _run_grid(cfg: BenchmarkConfig, bucket: str, key: str, file_size: int) -> list[CellResult]:
    """Run the full benchmark grid and return one CellResult per (concurrency, part_size) cell."""
    sample_bytes = min(cfg.sample_mb * 1024 * 1024, file_size)
    sample_mb_actual = sample_bytes / 1e6

    cells = [
        (c, p)
        for p in cfg.part_sizes_mb
        for c in cfg.concurrencies
    ]
    random.shuffle(cells)  # shuffle to avoid systematic order effects

    n_total = len(cells) * cfg.n_repeats
    run_idx = 0
    raw: dict[tuple[int, int], list[float]] = {}
    meta: dict[tuple[int, int], tuple[int, bool]] = {}  # (n_parts, single_part)
    errors: dict[tuple[int, int], str] = {}

    for concurrency, part_size_mb in cells:
        run_speeds = []
        last_error = None

        for repeat in range(cfg.n_repeats):
            run_idx += 1
            w = len(str(n_total))
            label = f"[{run_idx:{w}d}/{n_total}] concurrency={concurrency:2d}  part_size={part_size_mb:4d} MB"
            if cfg.n_repeats > 1:
                label += f"  (run {repeat + 1}/{cfg.n_repeats})"
            print(f"  {label} ... ", end="", flush=True)

            try:
                mbps, n_parts, single_part = _benchmark_cell(
                    cfg.anon, cfg.region, bucket, key, file_size,
                    concurrency, part_size_mb, sample_bytes,
                )
                run_speeds.append(mbps)
                meta[(concurrency, part_size_mb)] = (n_parts, single_part)
                print(f"{mbps:7.1f} MB/s")
            except Exception as exc:
                last_error = str(exc)
                print(f"FAILED: {exc}")

        if run_speeds:
            raw[(concurrency, part_size_mb)] = run_speeds
        else:
            errors[(concurrency, part_size_mb)] = last_error or "unknown error"

    results = []
    for concurrency in cfg.concurrencies:
        for part_size_mb in cfg.part_sizes_mb:
            key_pair = (concurrency, part_size_mb)
            if key_pair in raw:
                speeds = raw[key_pair]
                median_mbps = statistics.median(speeds)
                n_parts, single_part = meta[key_pair]
                results.append(CellResult(
                    concurrency=concurrency,
                    part_size_mb=part_size_mb,
                    throughput_mbps=median_mbps,
                    sample_mb=sample_mb_actual,
                    n_parts=n_parts,
                    single_part=single_part,
                ))
            else:
                results.append(CellResult(
                    concurrency=concurrency,
                    part_size_mb=part_size_mb,
                    throughput_mbps=float("nan"),
                    sample_mb=sample_mb_actual,
                    n_parts=0,
                    single_part=False,
                    error=errors.get(key_pair, "unknown error"),
                ))

    return results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _format_table(
    results: list[CellResult],
    concurrencies: list[int],
    part_sizes_mb: list[int],
    file_size_mb: float,
) -> str:
    """Return a formatted throughput table as a string."""
    lookup = {(r.concurrency, r.part_size_mb): r for r in results}

    # Column widths
    row_label_w = max(len("concurrency"), len(str(max(concurrencies))))
    col_w = 10  # width per data column

    # Header
    header_parts = [f"{'concurrency':>{row_label_w}}"]
    for p in part_sizes_mb:
        header_parts.append(f"part={p}MB".rjust(col_w))
    header = " | ".join(header_parts)

    sep = "-" * len(header)

    lines = [
        f"Throughput (MB/s) — sample={results[0].sample_mb:.0f} MB from {file_size_mb:.0f} MB file",
        "=" * len(header),
        header,
        sep,
    ]

    has_single_part = False
    best_mbps = max(
        (r.throughput_mbps for r in results if r.error is None),
        default=float("nan"),
    )

    for concurrency in concurrencies:
        row_parts = [f"{concurrency:>{row_label_w}}"]
        for part_size_mb in part_sizes_mb:
            cell = lookup.get((concurrency, part_size_mb))
            if cell is None or cell.error is not None:
                row_parts.append("ERR".rjust(col_w))
            else:
                marker = " *" if cell.single_part else ("  " if cell.throughput_mbps < best_mbps else " ←")
                has_single_part = has_single_part or cell.single_part
                row_parts.append(f"{cell.throughput_mbps:6.1f}{marker}".rjust(col_w))
        lines.append(" | ".join(row_parts))

    lines.append(sep)

    if has_single_part:
        lines.append(
            "* single-part: part_size >= sample size; only one request issued. "
            "Concurrency is irrelevant for these cells."
        )

    return "\n".join(lines)


def _recommend(results: list[CellResult]) -> tuple[int, int] | None:
    """Return (concurrency, part_size_mb) of the recommended configuration.

    Applies a 5% diminishing-returns guard: if a lower-concurrency cell is
    within 5% of the best throughput, prefer it to reduce DataLoader thread overhead.
    """
    valid = [r for r in results if r.error is None and not r.single_part]
    if not valid:
        # Fall back to including single-part results if nothing else is available
        valid = [r for r in results if r.error is None]
    if not valid:
        return None

    best = max(valid, key=lambda r: r.throughput_mbps)
    threshold = best.throughput_mbps * 0.95

    # Among cells within 5% of best, pick the one with lowest concurrency
    candidates = [r for r in valid if r.throughput_mbps >= threshold]
    recommended = min(candidates, key=lambda r: (r.concurrency, r.part_size_mb))
    return recommended.concurrency, recommended.part_size_mb


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def _save_csv(results: list[CellResult], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["concurrency", "part_size_mb", "throughput_mbps",
                        "sample_mb", "n_parts", "single_part", "error"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "concurrency": r.concurrency,
                "part_size_mb": r.part_size_mb,
                "throughput_mbps": f"{r.throughput_mbps:.2f}" if r.error is None else "",
                "sample_mb": f"{r.sample_mb:.1f}",
                "n_parts": r.n_parts,
                "single_part": r.single_part,
                "error": r.error or "",
            })


# ---------------------------------------------------------------------------
# Warm-up
# ---------------------------------------------------------------------------

def _warmup(anon: bool, region: str | None, bucket: str, key: str) -> None:
    """Download a small leading chunk to warm TCP connections and DNS."""
    print("  Warming up connection ... ", end="", flush=True)
    try:
        client = _make_client(anon, region, pool_size=4)
        _fetch_range(client, bucket, key, 0, min(4 * 1024 * 1024, 1))
        print("done")
    except Exception as exc:
        print(f"skipped ({exc})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark boto3 S3 download throughput for HelioNetCDFDataset.\n\n"
            "Tests combinations of s3_boto3_max_concurrency and s3_boto3_part_size_mb\n"
            "and prints a throughput table so you can pick the best settings for your\n"
            "connection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument(
        "s3_uri",
        help="S3 URI of a representative test file, ideally ~1 GB (e.g. an SDO NetCDF file).",
    )
    parser.add_argument(
        "--sample-mb",
        type=int,
        default=512,
        metavar="MB",
        help="Megabytes to download per benchmark run (default: 512). "
             "Capped at file size. Larger = more stable; smaller = faster.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a reduced 6-cell grid: concurrency=[1,4,16], part_size=[32,128]. "
             "Completes in ~2 min on a typical connection.",
    )
    parser.add_argument(
        "--concurrencies",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help="Concurrency values to test (default: 1 2 4 8 16).",
    )
    parser.add_argument(
        "--part-sizes-mb",
        nargs="+",
        type=int,
        default=None,
        metavar="MB",
        help="Part size values in MB to test (default: 16 64 128 256).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        metavar="N",
        help="Runs per grid cell; median throughput is reported (default: 3).",
    )
    parser.add_argument(
        "--anon",
        action="store_true",
        help="Use anonymous (unsigned) requests for publicly accessible buckets.",
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region override (default: from AWS_REGION env var or boto3 auto-detection).",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        metavar="PATH",
        help="Save results to a CSV file at PATH.",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the initial warm-up request.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Resolve grid
    if args.quick:
        concurrencies = QUICK_CONCURRENCIES
        part_sizes_mb = QUICK_PART_SIZES_MB
    else:
        concurrencies = args.concurrencies or list(FULL_CONCURRENCIES)
        part_sizes_mb = args.part_sizes_mb or list(FULL_PART_SIZES_MB)

    if args.repeats < 1:
        print("ERROR: --repeats must be at least 1.")
        sys.exit(1)
    if args.repeats == 1:
        print("WARNING: --repeats 1 gives unstable estimates. Use 3 or more for reliable results.\n")

    cfg = BenchmarkConfig(
        s3_uri=args.s3_uri,
        sample_mb=args.sample_mb,
        concurrencies=concurrencies,
        part_sizes_mb=part_sizes_mb,
        n_repeats=args.repeats,
        anon=args.anon,
        region=(args.region
                or os.environ.get("AWS_REGION")
                or os.environ.get("AWS_DEFAULT_REGION")
                or detect_ec2_region()),
        output_csv=args.output_csv,
        warmup=not args.no_warmup,
    )

    bucket, key = _parse_s3_uri(cfg.s3_uri)

    n_cells = len(concurrencies) * len(part_sizes_mb)
    print(f"S3 download benchmark")
    print(f"  URI:          {cfg.s3_uri}")
    print(f"  Concurrency:  {concurrencies}")
    print(f"  Part sizes:   {part_sizes_mb} MB")
    print(f"  Cells:        {n_cells}  (x{cfg.n_repeats} repeats = {n_cells * cfg.n_repeats} downloads)")
    print(f"  Sample:       up to {cfg.sample_mb} MB per download")
    print()

    # Probe file size
    print("Probing file size ... ", end="", flush=True)
    probe_client = _make_client(cfg.anon, cfg.region, pool_size=4)
    file_size = _probe_file_size(probe_client, bucket, key)
    file_size_mb = file_size / 1e6
    print(f"{file_size_mb:.1f} MB")

    if file_size_mb < 256:
        print(
            f"\nWARNING: File is only {file_size_mb:.0f} MB. SDO NetCDF files are typically ~1 GB. "
            "Part sizes larger than the file will test single-part behavior only.\n"
        )

    effective_sample_mb = min(cfg.sample_mb, file_size_mb)
    if effective_sample_mb < cfg.sample_mb:
        print(
            f"NOTE: --sample-mb {cfg.sample_mb} exceeds file size "
            f"({file_size_mb:.0f} MB). Sampling the full file.\n"
        )

    if cfg.region and detect_ec2_region():
        print(
            f"NOTE: Running on EC2 (region: {cfg.region}). "
            "Expected throughput: 500-1000+ MB/s when traffic stays on the AWS internal network.\n"
            "To ensure S3 traffic never leaves AWS, confirm that a VPC S3 Gateway Endpoint\n"
            "is attached to your VPC (AWS Console → VPC → Endpoints → filter by 'S3 Gateway').\n"
            "Without it, traffic may route through an internet or NAT gateway even within AWS.\n"
        )
    else:
        print(
            "NOTE: Over a regular internet connection, 20-150 MB/s is typical. "
            "On EC2 in the same AWS region as the bucket, expect 500-1000+ MB/s.\n"
        )

    # Warm up
    if cfg.warmup:
        _warmup(cfg.anon, cfg.region, bucket, key)
        print()

    # Run grid
    print(f"Running {n_cells * cfg.n_repeats} downloads (median of {cfg.n_repeats} reported per cell):\n")
    results = _run_grid(cfg, bucket, key, file_size)

    # Print table
    print()
    print(_format_table(results, concurrencies, part_sizes_mb, file_size_mb))

    # Recommend
    print()
    recommendation = _recommend(results)
    if recommendation is None:
        print("Could not produce a recommendation — all benchmark cells failed.")
    else:
        best_c, best_p = recommendation
        best_cell = next(
            r for r in results if r.concurrency == best_c and r.part_size_mb == best_p
        )
        print(f"Recommended settings ({best_cell.throughput_mbps:.1f} MB/s):")
        print()
        print(f"  s3_boto3_max_concurrency: {best_c}")
        print(f"  s3_boto3_part_size_mb:    {best_p}")
        print()
        print("Add these to the data section of your config_script.yaml:")
        print()
        print(f"    s3_boto3_max_concurrency: {best_c}")
        print(f"    s3_boto3_part_size_mb: {best_p}")

    # Save CSV
    if cfg.output_csv:
        _save_csv(results, cfg.output_csv)
        print(f"\nResults saved to: {cfg.output_csv}")


if __name__ == "__main__":
    main()
