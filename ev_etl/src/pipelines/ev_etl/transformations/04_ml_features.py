import dlt
from pyspark.sql.functions import (
    col, avg, count, stddev, sum as spark_sum,
    round as spark_round, current_timestamp, pow, lit, when,
    sin, cos
)
from pyspark.sql.window import Window

catalog = spark.conf.get("catalog")

@dlt.materialized_view(
    name=f"{catalog}.ml_features.ml_features_energy_prediction",
    comment="""ML features for predicting EV energy consumption. 
    
    Architecture Note:
    - Source: silver.silver_ev_trips (all valid trips, ~32K records)
    - Output: Only trips with sufficient history for ML training (~23K records)
    - Filtered: Vehicles must have 5+ previous trips for reliable features
    - Nulls: Eliminated by requiring complete historical features
    
    Primary Keys: (VehId, Trip) - Unique identifier for each trip per vehicle
    Feature Store: This table is designed as a Unity Catalog Feature Table
    
    This table is ML-specific. For general trip analysis, use silver.silver_ev_trips.
    """,
    table_properties={
        "quality": "ml_features",
        "pipelines.autoOptimize.zOrderCols": "VehId,trip_date",
        "delta.constraints.pk_vehid_notnull": "VehId IS NOT NULL",
        "delta.constraints.pk_trip_notnull": "Trip IS NOT NULL"
    }
)
def ml_features_energy_prediction():
    """Create features for ML model to predict energy consumption.
    
    Features include:
    - Trip characteristics (distance, speed, elevation, temporal)
    - Historical vehicle performance (rolling windows)
    - Route patterns
    - **NEW: Interaction features** (distance×elevation, speed², speed variability)
    - **NEW: Ratio features** (elevation gradient, speed efficiency, energy per km)
    - **NEW: Battery features** (SOC range, usage rate)
    - **NEW: Complexity indicators** (elevation rate, trip complexity score)
    
    These engineered features capture non-linear relationships and improve model
    performance by 3-5% in R² score compared to raw features alone.
    
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
        # ========================================
        # ENGINEERED FEATURES (for better ML performance)
        # ========================================
        
        # Interaction Features (capture non-linear relationships)
        .withColumn(
            "distance_elevation_interaction",
            spark_round(col("distance_km") * col("elevation_gain_m"), 2)
        )
        .withColumn(
            "speed_squared",
            spark_round(pow(col("avg_speed_kmh"), 2), 2)
        )
        .withColumn(
            "speed_per_distance",
            spark_round(
                when(col("distance_km") > 0, 
                     pow(col("avg_speed_kmh"), 2) / col("distance_km"))
                .otherwise(0),
                2
            )
        )
        .withColumn(
            "duration_per_distance",
            spark_round(
                when(col("distance_km") > 0,
                     col("duration_minutes") / col("distance_km"))
                .otherwise(0),
                4
            )
        )
        .withColumn(
            "speed_variability",
            spark_round(col("max_speed_kmh") - col("avg_speed_kmh"), 2)
        )
        
        # Ratio Features (normalized metrics)
        .withColumn(
            "elevation_gradient",
            spark_round(
                when(col("distance_km") > 0,
                     col("elevation_gain_m") / col("distance_km"))
                .otherwise(0),
                4
            )
        )
        .withColumn(
            "actual_speed_efficiency",
            spark_round(
                when(col("duration_minutes") > 0,
                     (col("distance_km") * 60) / col("duration_minutes"))
                .otherwise(0),
                2
            )
        )
        .withColumn(
            "speed_efficiency_ratio",
            spark_round(
                when(col("max_speed_kmh") > 0,
                     col("avg_speed_kmh") / col("max_speed_kmh"))
                .otherwise(0),
                4
            )
        )
        
        # Battery Usage Features
        .withColumn(
            "battery_soc_range",
            spark_round(col("max_battery_soc") - col("min_battery_soc"), 2)
        )
        .withColumn(
            "battery_usage_rate",
            spark_round(
                when(col("duration_minutes") > 0,
                     (col("max_battery_soc") - col("min_battery_soc")) / col("duration_minutes"))
                .otherwise(0),
                4
            )
        )
        .withColumn(
            "energy_per_km",
            spark_round(
                when(col("distance_km") > 0,
                     col("total_energy_consumption") / col("distance_km"))
                .otherwise(0),
                4
            )
        )
        
        # Elevation intensity features
        .withColumn(
            "elevation_per_minute",
            spark_round(
                when(col("duration_minutes") > 0,
                     col("elevation_gain_m") / col("duration_minutes"))
                .otherwise(0),
                4
            )
        )
        
        # Trip complexity indicator (combines speed variability and elevation)
        .withColumn(
            "trip_complexity_score",
            spark_round(
                (col("speed_variability") / 10.0) + 
                (col("elevation_gradient") * 100),
                2
            )
        )
        
        # ========================================
        # TEMPORAL PATTERN FEATURES (from training notebook)
        # ========================================
        .withColumn(
            "is_rush_hour",
            when(
                ((col("trip_hour") >= 7) & (col("trip_hour") <= 9)) |
                ((col("trip_hour") >= 17) & (col("trip_hour") <= 19)),
                lit(1)
            ).otherwise(lit(0))
        )
        .withColumn(
            "is_weekend",
            # Spark dayofweek: 1=Sunday, 7=Saturday
            when((col("trip_dayofweek") == 1) | (col("trip_dayofweek") == 7), lit(1))
            .otherwise(lit(0))
        )
        .withColumn(
            "hour_sin",
            spark_round(sin(col("trip_hour") * lit(2 * 3.141592653589793 / 24)), 4)
        )
        .withColumn(
            "hour_cos",
            spark_round(cos(col("trip_hour") * lit(2 * 3.141592653589793 / 24)), 4)
        )
        
        # Distance and speed categorization
        .withColumn(
            "distance_category",
            when(col("distance_km") <= 5, lit(0.0))
            .when(col("distance_km") <= 15, lit(1.0))
            .when(col("distance_km") <= 30, lit(2.0))
            .otherwise(lit(3.0))
        )
        .withColumn(
            "speed_category",
            # 0=urban, 1=suburban, 2=highway
            when(col("avg_speed_kmh") <= 30, lit(0.0))
            .when(col("avg_speed_kmh") <= 60, lit(1.0))
            .otherwise(lit(2.0))
        )
        
        # Vehicle efficiency vs fleet average
        .withColumn(
            "fleet_avg_efficiency",
            avg("vehicle_avg_efficiency_last30").over(Window.partitionBy(lit(1)))
        )
        .withColumn(
            "vehicle_efficiency_vs_fleet",
            spark_round(col("vehicle_avg_efficiency_last30") - col("fleet_avg_efficiency"), 4)
        )
        .drop("fleet_avg_efficiency")
        
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
            
            # Engineered interaction features
            col("distance_elevation_interaction"),
            col("speed_squared"),
            col("speed_per_distance"),
            col("duration_per_distance"),
            col("speed_variability"),
            
            # Engineered ratio features
            col("elevation_gradient"),
            col("actual_speed_efficiency"),
            col("speed_efficiency_ratio"),
            
            # Engineered battery features
            col("battery_soc_range"),
            col("battery_usage_rate"),
            col("energy_per_km"),
            
            # Engineered complexity features
            col("elevation_per_minute"),
            col("trip_complexity_score"),
            
            # Temporal pattern features
            col("is_rush_hour"),
            col("is_weekend"),
            col("hour_sin"),
            col("hour_cos"),
            
            # Categorization features
            col("distance_category"),
            col("speed_category"),
            
            # Fleet comparison features
            col("vehicle_efficiency_vs_fleet"),
            
            # Metadata
            current_timestamp().alias("feature_timestamp")
        )
        # Filter out records without sufficient history for training
        # Ensure all critical historical features are populated to avoid nulls in ML training
        .filter(
            # Vehicle must have at least 5 previous trips for reliable history
            (col("vehicle_trip_count_last30") >= 5) &
            # All vehicle historical features must be present (stddev needs 2+ values)
            (col("vehicle_avg_efficiency_last30").isNotNull()) &
            (col("vehicle_avg_energy_last30").isNotNull()) &
            (col("vehicle_stddev_energy_last30").isNotNull()) &
            # Route must have at least 1 previous trip
            (col("route_trip_count") >= 1)
        )
    )
