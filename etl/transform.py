from __future__ import annotations

import logging
from typing import Optional, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
import math

logger = logging.getLogger(__name__)


UNIFIED_COLS = [
    "source",
    "trip_id",
    "route_id",
    "operating_day",
    "start_ts",
    "end_ts",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
    "distance_km",
    "duration_min",
    "avg_speed_kmh",
    "passenger_count",
    "delay_min",
    "delay_reason",
    "weather_main",
    "weather_description",
    "hour",
    "day_of_week",
]


def _has_col(df: DataFrame, name: str) -> bool:
    return name in df.columns


def transform_addis_trips(df: DataFrame) -> DataFrame:
    """
    Normalize Addis trip CSVs into the unified schema.

    Handles two observed schemas:
      Schema A (trip_data.csv):
        Date, Beginning Time, End Time, Mileage,
        Initial latitude, Initial longitude, Final latitude, Final longitude
      Schema B (final_data.csv):
        DayofWeek, TimeRange, Beginning Time, Mileage,
        Initial latitude, Initial longitude, Final latitude, Final longitude,
        End Time, total_time, Avg_Speed, Month
    """

    # Trim column names that include trailing spaces in the raw dataset.
    for c in df.columns:
        if c != c.strip():
            df = df.withColumnRenamed(c, c.strip())

    # Standardize coordinate columns (handle trailing-space variants)
    for raw, std in [
        ("Initial latitude", "start_lat"),
        ("Initial latitude ", "start_lat"),
        ("Initial longitude", "start_lon"),
        ("Initial longitude ", "start_lon"),
        ("Final latitude", "end_lat"),
        ("Final latitude ", "end_lat"),
        ("Final longitude", "end_lon"),
        ("Final longitude ", "end_lon"),
    ]:
        if raw in df.columns and std not in df.columns:
            df = df.withColumnRenamed(raw, std)

    # Rename time columns to neutral names
    if "Beginning Time" in df.columns:
        df = df.withColumnRenamed("Beginning Time", "begin_time_str")
    if "End Time" in df.columns:
        df = df.withColumnRenamed("End Time", "end_time_str")

    begin_str = F.col("begin_time_str").cast("string")
    end_str = F.col("end_time_str").cast("string")

    # ---- operating_day ----
    # Schema A has a `Date` column; Schema B has `Month` + `DayofWeek` (no calendar date).
    if _has_col(df, "Date"):
        date_str = F.col("Date").cast("string")
        op_ts = F.coalesce(
            F.try_to_timestamp(date_str, F.lit("MM/dd/yyyy")),
            F.try_to_timestamp(date_str, F.lit("M/d/yyyy")),
            F.try_to_timestamp(date_str, F.lit("MM/dd/yyyy HH:mm:ss")),
            F.try_to_timestamp(date_str, F.lit("M/d/yyyy HH:mm:ss")),
            F.try_to_timestamp(date_str, F.lit("yyyy-MM-dd")),
            F.try_to_timestamp(date_str),
        )
        df = df.withColumn("operating_day", F.to_date(op_ts))
        # Build timestamps by combining date + time-of-day strings
        start_ts = F.coalesce(
            F.try_to_timestamp(F.concat_ws(" ", date_str, begin_str), F.lit("MM/dd/yyyy HH:mm:ss")),
            F.try_to_timestamp(F.concat_ws(" ", date_str, begin_str), F.lit("M/d/yyyy HH:mm:ss")),
            F.try_to_timestamp(begin_str),
            F.try_to_timestamp(begin_str, F.lit("M/d/yyyy H:mm:ss")),
            F.try_to_timestamp(begin_str, F.lit("MM/dd/yyyy H:mm:ss")),
        )
        end_ts = F.coalesce(
            F.try_to_timestamp(F.concat_ws(" ", date_str, end_str), F.lit("MM/dd/yyyy HH:mm:ss")),
            F.try_to_timestamp(F.concat_ws(" ", date_str, end_str), F.lit("M/d/yyyy HH:mm:ss")),
            F.try_to_timestamp(end_str),
            F.try_to_timestamp(end_str, F.lit("M/d/yyyy H:mm:ss")),
            F.try_to_timestamp(end_str, F.lit("MM/dd/yyyy H:mm:ss")),
        )
        df = df.withColumn("start_ts", start_ts)
        df = df.withColumn("end_ts", end_ts)
    else:
        # Schema B (final_data.csv): no recoverable wall-clock timestamps.
        # `Beginning Time` and `End Time` are minute-of-hour integers with no hour context.
        # `total_time` (hours) and `Avg_Speed` (km/h) are pre-computed and reliable.
        # Synthesise a proxy date: 2026-<Month>-01 anchored to DayofWeek offset.
        month_col = F.col("Month").cast("int")
        dow_col = F.col("DayofWeek").cast("int")  # 1=Mon..7=Sun (ISO-style in source)

        # Build a proxy operating_day: first day of the given month in 2026
        df = df.withColumn(
            "operating_day",
            F.to_date(
                F.concat_ws("-", F.lit("2026"), F.lpad(month_col.cast("string"), 2, "0"), F.lit("01"))
            ),
        )
        # Use total_time (hours) for duration; synthesise start_ts as midnight of operating_day
        df = df.withColumn("start_ts", F.to_timestamp(F.col("operating_day")))
        df = df.withColumn(
            "end_ts",
            F.col("start_ts") + F.expr("INTERVAL 1 SECOND") * (F.col("total_time").cast("double") * 3600).cast("long"),
        )

    # If end time is earlier than start time (crossing midnight), add 1 day.
    df = df.withColumn(
        "end_ts",
        F.when(F.col("end_ts") < F.col("start_ts"), F.col("end_ts") + F.expr("INTERVAL 1 DAY")).otherwise(F.col("end_ts")),
    )

    df = df.withColumn("distance_km", F.col("Mileage").cast("double") * F.lit(1.60934))

    # Schema B has pre-computed total_time (hours) and Avg_Speed (km/h) — use them directly
    if _has_col(df, "total_time"):
        df = df.withColumn("duration_min", F.col("total_time").cast("double") * 60.0)
        df = df.withColumn("avg_speed_kmh", F.col("Avg_Speed").cast("double"))
    else:
        df = df.withColumn("duration_min", (F.col("end_ts").cast("long") - F.col("start_ts").cast("long")) / 60.0)
        df = df.withColumn(
            "avg_speed_kmh",
            F.when(F.col("duration_min") > 0, F.col("distance_km") / (F.col("duration_min") / 60.0)).otherwise(F.lit(None)),
        )

    # Expected travel time based on nominal 25 km/h (traffic baseline)
    expected_min = F.when(F.col("distance_km").isNotNull(), (F.col("distance_km") / F.lit(25.0)) * 60.0).otherwise(F.lit(None))
    df = df.withColumn("delay_min", F.greatest(F.lit(0.0), F.col("duration_min") - expected_min))

    df = df.withColumn("route_id", F.lit(None).cast("string"))
    df = df.withColumn("passenger_count", F.lit(None).cast("int"))

    # Deterministic trip_id — use whichever date/time columns are available
    date_key = F.col("Date").cast("string") if _has_col(df, "Date") else F.col("operating_day").cast("string")
    df = df.withColumn(
        "trip_id",
        F.sha2(
            F.concat_ws(
                "|",
                date_key,
                F.col("begin_time_str").cast("string"),
                F.col("end_time_str").cast("string"),
                F.col("start_lat").cast("string"),
                F.col("start_lon").cast("string"),
                F.col("end_lat").cast("string"),
                F.col("end_lon").cast("string"),
            ),
            256,
        ),
    )

    df = df.withColumn("hour", F.hour(F.col("start_ts")))
    # ISO weekday (Mon=1..Sun=7).
    # Schema B has DayofWeek directly; Schema A derives it from operating_day.
    if _has_col(df, "DayofWeek"):
        df = df.withColumn("day_of_week", F.col("DayofWeek").cast("int"))
    else:
        df = df.withColumn("day_of_week", F.expr("((dayofweek(operating_day)+5)%7)+1").cast("int"))

    df = df.withColumn("source", F.lit("addis_csv"))
df = df.select(
    
        "source",
        "trip_id",
        "route_id",
        "operating_day",
        "start_ts",
        "end_ts",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "distance_km",
        "duration_min",
        "avg_speed_kmh",
        "passenger_count",
        "delay_min",
        "hour",
        "day_of_week",
    )

    return df


def transform_passengers_parquet(df: DataFrame) -> DataFrame:
    """
    Normalize the Kaggle passenger parquet into the unified schema.

    Observed schema:
      operating_day, line_id, trip_departure, trip_arrival, departure, arrival, passengers, vehicle_seats, ...
    """

    # Source parquet has `operating_day` as a string with varying formats
    # (e.g., "2021-01-02", "13.02.22 00:00:00"). Use tolerant parsing.
    op_str = F.col("operating_day").cast("string")
    op_ts = F.coalesce(
        F.try_to_timestamp(op_str, F.lit("yyyy-MM-dd")),
        F.try_to_timestamp(op_str, F.lit("yyyy-MM-dd HH:mm:ss")),
        F.try_to_timestamp(op_str, F.lit("dd.MM.yy HH:mm:ss")),
        F.try_to_timestamp(op_str),
    )
    df = df.withColumn("operating_day", F.to_date(op_ts))
    df = df.withColumn("route_id", F.col("line_id").cast("string"))

    # Times are seconds since midnight (assumed); convert to timestamps by adding seconds to operating_day.
    df = df.withColumn("start_ts", F.to_timestamp("operating_day") + F.expr("INTERVAL 1 SECOND") * F.col("trip_departure").cast("int"))
    df = df.withColumn("end_ts", F.to_timestamp("operating_day") + F.expr("INTERVAL 1 SECOND") * F.col("trip_arrival").cast("int"))

    df = df.withColumn("duration_min", (F.col("trip_arrival").cast("double") - F.col("trip_departure").cast("double")) / 60.0)
    df = df.withColumn("distance_km", F.lit(None).cast("double"))
    df = df.withColumn("avg_speed_kmh", F.lit(None).cast("double"))
    df = df.withColumn("passenger_count", F.col("passengers").cast("int"))

    # Delay metrics: compare actual departure/arrival (departure/arrival) vs scheduled trip_*.
    df = df.withColumn("delay_min", F.greatest(F.lit(0.0), (F.col("departure") - F.col("trip_departure")) / 60.0))

    df = df.withColumn("start_lat", F.lit(None).cast("double"))
    df = df.withColumn("start_lon", F.lit(None).cast("double"))
    df = df.withColumn("end_lat", F.lit(None).cast("double"))
    df = df.withColumn("end_lon", F.lit(None).cast("double"))

    # Stable trip_id within a day/route/time.
    df = df.withColumn(
        "trip_id",
        F.sha2(
            F.concat_ws(
                "|",
                F.col("operating_day").cast("string"),
                F.col("route_id").cast("string"),
                F.col("trip_departure").cast("string"),
                F.col("trip_arrival").cast("string"),
                F.col("stop_id").cast("string"),
            ),
            256,
        ),
    )

    df = df.withColumn("hour", F.hour(F.col("start_ts")))
    # ISO weekday (Mon=1..Sun=7) without relying on datetime patterns.
    df = df.withColumn("day_of_week", F.expr("((dayofweek(operating_day)+5)%7)+1").cast("int"))
    df = df.withColumn("source", F.lit("passengers_parquet"))

    df = df.select(
        "source",
        "trip_id",
        "route_id",
        "operating_day",
        "start_ts",
        "end_ts",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "distance_km",
        "duration_min",
        "avg_speed_kmh",
        "passenger_count",
        "delay_min",
        "hour",
        "day_of_week",
    )
    return df


def attach_weather_and_delay_reason(
    trips: DataFrame,
    *,
    api_key: Optional[str] = None,
    base_url: str = "https://api.openweathermap.org/data/2.5",
    cache_path: Optional[str] = None,
    max_api_lookups: int = 200,
) -> DataFrame:
    """
    Attach weather fields and derive delay_reason.

    Strategy:
      - If `api_key` is provided, do a *bounded* number of lookups on the driver
        for distinct (rounded lat/lon, hour) buckets and join back.
      - Fallback to a deterministic mock rule by hour.
    """

    spark = trips.sparkSession

    # Bucket keys for weather enrichment
    trips = trips.withColumn("w_lat", F.round(F.col("start_lat"), 2))
    trips = trips.withColumn("w_lon", F.round(F.col("start_lon"), 2))
    trips = trips.withColumn("w_epoch_hour", (F.unix_timestamp(F.col("start_ts")) / 3600).cast("bigint"))

    enriched = False
    if api_key and cache_path:
        try:
            from etl.weather import get_weather_snapshot  # noqa: WPS433 (runtime import is intentional)
            from pathlib import Path

            keys = (
                trips.select("w_lat", "w_lon", "w_epoch_hour")
                .where(F.col("w_lat").isNotNull() & F.col("w_lon").isNotNull() & F.col("w_epoch_hour").isNotNull())
                .dropDuplicates(["w_lat", "w_lon", "w_epoch_hour"])
                .limit(int(max_api_lookups))
                .collect()
            )

            rows = []
            for r in keys:
                snap = get_weather_snapshot(
                    lat=float(r["w_lat"]),
                    lon=float(r["w_lon"]),
                    epoch_hour=int(r["w_epoch_hour"]),
                    api_key=api_key,
                    base_url=base_url,
                    cache_path=Path(cache_path),
                    allow_mock=True,
                )
                rows.append((float(r["w_lat"]), float(r["w_lon"]), int(r["w_epoch_hour"]), snap.weather_main, snap.weather_description))

            if rows:
                weather_df = spark.createDataFrame(rows, ["w_lat", "w_lon", "w_epoch_hour", "weather_main", "weather_description"])
                trips = trips.join(weather_df, on=["w_lat", "w_lon", "w_epoch_hour"], how="left")
                enriched = True
        except Exception:
            logger.exception("Weather API enrichment failed; falling back to mock enrichment.")

    if not enriched:
        trips = trips.withColumn(
            "weather_main",
            F.when(F.col("hour").between(16, 20), F.lit("Rain"))
            .when(F.col("hour").between(7, 9), F.lit("Clouds"))
            .otherwise(F.lit("Clear")),
        )
        trips = trips.withColumn(
            "weather_description",
            F.when(F.col("weather_main") == "Rain", F.lit("light rain"))
            .when(F.col("weather_main") == "Clouds", F.lit("broken clouds"))
            .otherwise(F.lit("clear sky")),
        )

    trips = trips.withColumn(
        "delay_reason",
        F.when((F.col("delay_min") > 0) & (F.col("weather_main").isin("Rain", "Thunderstorm", "Snow", "Drizzle")), F.lit("Weather"))
        .when(F.col("delay_min") > 0, F.lit("Traffic"))
        .otherwise(F.lit(None).cast("string")),
    )
    trips = trips.drop("w_lat", "w_lon", "w_epoch_hour")
    return trips


def unify_trips(
    addis: DataFrame,
    passengers: DataFrame,
    *,
    weather_api_key: Optional[str] = None,
    weather_base_url: str = "https://api.openweathermap.org/data/2.5",
    weather_cache_path: Optional[str] = None,
) -> DataFrame:
    """
    Union trips into the unified dataset, clean nulls/duplicates, and add analytics helpers.
    """

    trips = addis.unionByName(passengers, allowMissingColumns=True)

    # Clean: remove impossible durations, drop dup trip_id
    trips = trips.filter((F.col("duration_min").isNull()) | (F.col("duration_min") >= 0))
    trips = trips.dropDuplicates(["trip_id"])

    trips = attach_weather_and_delay_reason(
        trips,
        api_key=weather_api_key,
        base_url=weather_base_url,
        cache_path=weather_cache_path,
    )

    # Peak hour detection: mark the top ~25% busiest hours by trip count.
    #
    # `hour` has very low cardinality (<=24). Instead of a window (which can
    # collapse to a single partition and be memory-heavy), compute the 75th
    # percentile threshold on the driver and join a tiny mapping table back.
    hour_counts = trips.groupBy("hour").agg(F.count("*").alias("trip_count"))
    rows = [r for r in hour_counts.where(F.col("hour").isNotNull()).collect()]
    if rows:
        counts_sorted = sorted((int(r["trip_count"]) for r in rows))
        idx = int(math.floor(0.75 * (len(counts_sorted) - 1)))
        threshold = counts_sorted[idx]
        mapping = [(int(r["hour"]), bool(int(r["trip_count"]) >= threshold)) for r in rows]
        hour_flags = trips.sparkSession.createDataFrame(mapping, ["hour", "is_peak_hour_by_hour"])
        trips = trips.join(hour_flags, on="hour", how="left")
    else:
        trips = trips.withColumn("is_peak_hour_by_hour", F.lit(False))
    trips = trips.withColumn("is_peak_hour", F.coalesce(F.col("is_peak_hour_by_hour"), F.lit(False)))
    trips = trips.drop("is_peak_hour_by_hour")

    # Ensure all unified columns exist (plus is_peak_hour)
    for c in UNIFIED_COLS:
        if c not in trips.columns:
            trips = trips.withColumn(c, F.lit(None))

    return trips


def build_analytics_summary(trips: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Build `routes` and `analytics_summary` tables.
    """

    routes = (
        trips.select("route_id")
        .where(F.col("route_id").isNotNull())
        .dropDuplicates(["route_id"])
        .withColumn("route_name", F.concat(F.lit("Route "), F.col("route_id")))
    )
    summary = (
        trips.groupBy("operating_day", "hour", "source")
        .agg(
            F.count("*").alias("trip_count"),
            F.sum(F.coalesce(F.col("passenger_count"), F.lit(0))).alias("total_passengers"),
            F.avg(F.col("delay_min")).alias("avg_delay_min"),
            F.sum(F.when(F.col("is_peak_hour") == True, 1).otherwise(0)).alias("peak_hour_trip_count"),
        )
        .withColumn("created_at", F.current_timestamp())
    )

    return routes, summary
    
