import dlt
from pyspark.sql.functions import (
    col, avg, count, stddev, sum as spark_sum,
    round as spark_round, current_timestamp
)
from pyspark.sql.window import Window

catalog = spark.conf.get("catalog")

@dlt.materialized_view(
    name=f"{catalog}.ml_features.ml_features_energy_prediction",
    comment="ML features for predicting EV energy consumption. Includes trip characteristics and historical patterns.",
    table_properties={
        "quality": "ml_features",
        "pipelines.autoOptimize.zOrderCols": "VehId,trip_date"
    }
)
def ml_features_energy_prediction():
    """Create features for ML model to predict energy consumption.
    
    Features include:
    - Trip characteristics (distance, speed, elevation, temporal)
    - Historical vehicle performance (rolling windows)
    - Route patterns
    
    Target variable: total_energy_consumption (to predict)
    """
    # Window specs for historical features using rowsBetween
    vehicle_window = (
        Window
        .partitionBy("VehId")
        .orderBy(col("start_timestamp_ms").cast("long"))
        .rowsBetween(-30, -1)  # Last 30 trips per vehicle
    )
    
    route_window = (
        Window
        .partitionBy("origin_lat_rounded", "origin_lon_rounded", 
                    "dest_lat_rounded", "dest_lon_rounded")
        .orderBy(col("start_timestamp_ms").cast("long"))
        .rowsBetween(-10, -1)  # Last 10 trips on similar routes
    )
    
    df = spark.read.table(f"{catalog}.silver.silver_ev_trips")
    
    # Add rounded coordinates for route grouping
    df = (
        df
        .withColumn("origin_lat_rounded", spark_round(col("origin_lat"), 2))
        .withColumn("origin_lon_rounded", spark_round(col("origin_lon"), 2))
        .withColumn("dest_lat_rounded", spark_round(col("destination_lat"), 2))
        .withColumn("dest_lon_rounded", spark_round(col("destination_lon"), 2))
    )
    
    return (
        df
        # Historical vehicle features (last 30 trips)
        .withColumn(
            "vehicle_avg_efficiency_last30",
            spark_round(avg("efficiency_km_per_energy").over(vehicle_window), 4)
        )
        .withColumn(
            "vehicle_avg_energy_last30",
            spark_round(avg("total_energy_consumption").over(vehicle_window), 2)
        )
        .withColumn(
            "vehicle_stddev_energy_last30",
            spark_round(stddev("total_energy_consumption").over(vehicle_window), 2)
        )
        .withColumn(
            "vehicle_avg_distance_last30",
            spark_round(avg("distance_km").over(vehicle_window), 2)
        )
        .withColumn(
            "vehicle_trip_count_last30",
            count("Trip").over(vehicle_window)
        )
        # Historical route features
        .withColumn(
            "route_avg_energy_last10",
            spark_round(avg("total_energy_consumption").over(route_window), 2)
        )
        .withColumn(
            "route_avg_duration_last10",
            spark_round(avg("duration_minutes").over(route_window), 2)
        )
        .withColumn(
            "route_avg_speed_last10",
            spark_round(avg("avg_speed_kmh").over(route_window), 2)
        )
        .withColumn(
            "route_trip_count",
            count("Trip").over(route_window)
        )
        # Select features in logical order
        .select(
            # Identifiers
            col("DayNum"),
            col("VehId"),
            col("Trip"),
            col("trip_date"),
            col("start_timestamp_ms"),
            
            # Target variable
            col("total_energy_consumption").alias("target_energy_consumption"),
            
            # Primary trip features
            col("distance_km"),
            col("duration_minutes"),
            col("avg_speed_kmh"),
            col("max_speed_kmh"),
            
            # Battery features
            col("avg_battery_soc"),
            col("min_battery_soc"),
            col("max_battery_soc"),
            
            # Elevation features
            col("avg_elevation_m"),
            col("elevation_gain_m"),
            
            # Location features (rounded for privacy/grouping)
            col("origin_lat_rounded"),
            col("origin_lon_rounded"),
            col("dest_lat_rounded"),
            col("dest_lon_rounded"),
            
            # Temporal features
            col("trip_year"),
            col("trip_month"),
            col("trip_dayofweek"),
            col("trip_hour"),
            
            # Historical vehicle features (30 trips)
            col("vehicle_avg_efficiency_last30"),
            col("vehicle_avg_energy_last30"),
            col("vehicle_stddev_energy_last30"),
            col("vehicle_avg_distance_last30"),
            col("vehicle_trip_count_last30"),
            
            # Historical route features
            col("route_avg_energy_last10"),
            col("route_avg_duration_last10"),
            col("route_avg_speed_last10"),
            col("route_trip_count"),
            
            # Metadata
            current_timestamp().alias("feature_timestamp")
        )
        # Filter out records without sufficient history for training
        .filter(
            (col("vehicle_trip_count_last30").isNotNull()) &
            (col("route_trip_count") >= 1)
        )
    )
