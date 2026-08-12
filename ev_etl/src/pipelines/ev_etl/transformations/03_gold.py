import dlt
from pyspark.sql.functions import (
    col, avg, sum as spark_sum, count, max as spark_max, min as spark_min,
    stddev, round as spark_round, current_timestamp, percentile_approx
)

catalog = spark.conf.get("catalog")

@dlt.materialized_view(
    name=f"{catalog}.gold.gold_vehicle_daily_stats",
    comment="Daily vehicle statistics from aggregated trip data. Useful for fleet monitoring and performance analysis.",
    table_properties={
        "quality": "gold",
        "pipelines.autoOptimize.zOrderCols": "VehId,trip_date"
    }
)
def gold_vehicle_daily_stats():
    """Aggregate daily statistics by vehicle.
    
    Provides summary metrics for:
    - Daily trip patterns per vehicle
    - Energy consumption trends
    - Efficiency and performance metrics
    """
    return (
        spark.read.table("silver_ev_trips")
            .groupBy("VehId", "trip_date")
            .agg(
                count("Trip").alias("total_trips"),
                spark_sum("distance_km").alias("total_distance_km"),
                spark_sum("total_energy_consumption").alias("total_energy_consumption"),
                spark_sum("duration_minutes").alias("total_duration_minutes"),
                avg("efficiency_km_per_energy").alias("avg_efficiency"),
                avg("avg_speed_kmh").alias("avg_speed_kmh"),
                avg("avg_battery_soc").alias("avg_battery_soc"),
                spark_max("distance_km").alias("max_trip_distance_km"),
                spark_min("distance_km").alias("min_trip_distance_km"),
                stddev("distance_km").alias("stddev_distance_km"),
                spark_max("max_speed_kmh").alias("max_speed_observed"),
                avg("elevation_gain_m").alias("avg_elevation_gain_m")
            )
            .withColumn("total_distance_km", spark_round(col("total_distance_km"), 2))
            .withColumn("total_energy_consumption", spark_round(col("total_energy_consumption"), 2))
            .withColumn("avg_efficiency", spark_round(col("avg_efficiency"), 4))
            .withColumn("avg_speed_kmh", spark_round(col("avg_speed_kmh"), 2))
            .withColumn("avg_battery_soc", spark_round(col("avg_battery_soc"), 2))
            .withColumn("processing_timestamp", current_timestamp())
    )


@dlt.materialized_view(
    name="gold_route_patterns",
    comment="Route-level analysis with energy consumption patterns. Groups trips by approximate origin-destination pairs.",
    table_properties={
        "quality": "gold"
    }
)
def gold_route_patterns():
    """Analyze route patterns to identify energy consumption trends.
    
    Groups by rounded GPS coordinates to identify common routes.
    Useful for:
    - Route optimization
    - Charging station planning
    - Energy consumption prediction
    """
    return (
        spark.read.table("silver_ev_trips")
            # Round coordinates to ~1km precision for route grouping
            .withColumn("origin_lat_rounded", spark_round(col("origin_lat"), 2))
            .withColumn("origin_lon_rounded", spark_round(col("origin_lon"), 2))
            .withColumn("dest_lat_rounded", spark_round(col("destination_lat"), 2))
            .withColumn("dest_lon_rounded", spark_round(col("destination_lon"), 2))
            .groupBy(
                "origin_lat_rounded", "origin_lon_rounded",
                "dest_lat_rounded", "dest_lon_rounded"
            )
            .agg(
                count("Trip").alias("trip_count"),
                avg("distance_km").alias("avg_distance_km"),
                avg("total_energy_consumption").alias("avg_energy_consumption"),
                avg("duration_minutes").alias("avg_duration_minutes"),
                avg("efficiency_km_per_energy").alias("avg_efficiency"),
                avg("avg_speed_kmh").alias("avg_speed_kmh"),
                avg("elevation_gain_m").alias("avg_elevation_gain_m"),
                stddev("total_energy_consumption").alias("stddev_energy"),
                count(col("VehId").cast("string")).alias("unique_vehicles")
            )
            .filter(col("trip_count") >= 3)  # Only routes with multiple observations
            .withColumn("avg_distance_km", spark_round(col("avg_distance_km"), 2))
            .withColumn("avg_energy_consumption", spark_round(col("avg_energy_consumption"), 2))
            .withColumn("avg_efficiency", spark_round(col("avg_efficiency"), 4))
            .withColumn("avg_speed_kmh", spark_round(col("avg_speed_kmh"), 2))
            .withColumn("processing_timestamp", current_timestamp())
    )


@dlt.materialized_view(
    name="gold_temporal_patterns",
    comment="Temporal energy consumption patterns by hour and day of week",
    table_properties={
        "quality": "gold"
    }
)
def gold_temporal_patterns():
    """Analyze temporal patterns in energy consumption.
    
    Identifies:
    - Peak usage hours
    - Day-of-week patterns
    - Seasonal trends
    """
    return (
        spark.read.table("silver_ev_trips")
            .groupBy("trip_dayofweek", "trip_hour")
            .agg(
                count("Trip").alias("trip_count"),
                avg("distance_km").alias("avg_distance_km"),
                avg("total_energy_consumption").alias("avg_energy_consumption"),
                avg("avg_speed_kmh").alias("avg_speed_kmh"),
                avg("efficiency_km_per_energy").alias("avg_efficiency"),
                stddev("total_energy_consumption").alias("stddev_energy")
            )
            .withColumn("avg_distance_km", spark_round(col("avg_distance_km"), 2))
            .withColumn("avg_energy_consumption", spark_round(col("avg_energy_consumption"), 2))
            .withColumn("avg_efficiency", spark_round(col("avg_efficiency"), 4))
            .withColumn("processing_timestamp", current_timestamp())
    )
