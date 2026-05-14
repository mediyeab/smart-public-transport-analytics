from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from pyspark.sql import DataFrame, SparkSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractedSources:
    # Schema A CSVs: have a `Date` column (e.g. trip_data.csv)
    addis_trips_csv: List[Path]
    # Schema B CSVs: have `DayofWeek`/`Month`/`TimeRange` but no `Date` (e.g. final_data.csv)
    addis_final_csv: List[Path]
    passengers_parquet: List[Path]


# Columns that identify Schema B (final_data.csv style)
_SCHEMA_B_COLS = {"DayofWeek", "TimeRange", "Month"}
# Columns that identify Schema A (trip_data.csv style)
_SCHEMA_A_COLS = {"Date"}

# GTFS-RT parquet files are not in the passenger schema; exclude them.
_GTFS_PARQUET_PATTERNS = ("gtfsrt", "gtfs_rt", "compacted")


def _is_gtfs_parquet(path: Path) -> bool:
    return any(pat in path.name.lower() for pat in _GTFS_PARQUET_PATTERNS)


def discover_sources(raw_dir: Path) -> ExtractedSources:
    """
    Discover raw inputs under data/raw.

    Separates:
      - Schema A CSVs (trip_data.csv style, have `Date` column)
      - Schema B CSVs (final_data.csv style, have `DayofWeek`/`Month`/`TimeRange`)
      - Passenger parquet: `public_transport.parquet` (excludes GTFS-RT parquets)
    """

    csv_candidates = sorted(raw_dir.rglob("*.csv"))
schema_a: List[Path] = []
    schema_b: List[Path] = []

    
    for p in csv_candidates:
        # Peek at the header row to determine schema
        try:
            with p.open(encoding="utf-8", errors="replace") as fh:
                header = fh.readline().strip()
            cols = {c.strip() for c in header.split(",")}
            if _SCHEMA_B_COLS.issubset(cols):
                schema_b.append(p)
            elif _SCHEMA_A_COLS.issubset(cols):
                schema_a.append(p)
            else:
                # Fallback: classify by filename
                if "final" in p.name.lower():
                    schema_b.append(p)
                else:
                    schema_a.append(p)
        except Exception:
            logger.warning("Could not peek at CSV header for %s; skipping.", p)

    parquet_candidates = sorted(raw_dir.rglob("*.parquet"))
    passengers = [p for p in parquet_candidates if p.name.lower() == "public_transport.parquet"]
    if not passengers:
        # Skip GTFS-RT files because their structure is different
        passengers = [p for p in parquet_candidates if not _is_gtfs_parquet(p)]

    logger.info(
        "Discovered %s Schema-A CSVs, %s Schema-B CSVs, %s parquet files",
        len(schema_a), len(schema_b), len(passengers),
    )
    return ExtractedSources(
        addis_trips_csv=schema_a,
        addis_final_csv=schema_b,
        passengers_parquet=passengers,
    )


def _paths_to_str(paths: Iterable[Path]) -> List[str]:
    return [str(p) for p in paths]


def read_addis_trips_csv(spark: SparkSession, csv_paths: List[Path]) -> DataFrame:
    """Read Schema-A CSVs (have a `Date` column)."""
    if not csv_paths:
        raise FileNotFoundError("No Schema-A Addis trips CSV files found under data/raw.")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(_paths_to_str(csv_paths))
    )
    return df


def read_addis_final_csv(spark: SparkSession, csv_paths: List[Path]) -> DataFrame:
    """Read Schema-B CSVs (DayofWeek/Month/TimeRange style, e.g. final_data.csv)."""
    if not csv_paths:
        raise FileNotFoundError("No Schema-B Addis final CSV files found under data/raw.")

    df = (
        spark.read.option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(_paths_to_str(csv_paths))
    )
    return df
# Reads passenger parquet files into a Spark DataFrame after validating file existence.

def read_passengers_parquet(spark: SparkSession, parquet_paths: List[Path]) -> DataFrame:
    if not parquet_paths:
        raise FileNotFoundError("No passenger parquet files found under data/raw.")
    df = spark.read.parquet(*_paths_to_str(parquet_paths))
    return df

