import dlt
from pyspark.sql.functions import (
    col, min as spark_min, max as spark_max, avg, sum as spark_sum,
    count, first, last, round as spark_round, radians, sin, cos, sqrt, atan2,
    from_unixtime, to_timestamp, year, month, dayofweek, hour, current_timestamp
)

catalog = spark.conf.get("catalog")

@dlt.materialized_view(
    name=f"{catalog}.silver.silver_ev_trips",
    comment="Aggregated trip records from telemetry data. Each row represents one complete trip with calculated metrics.",
    table_properties={
        "quality": "silver",
        "pipelines.autoOptimize.zOrderCols": "VehId,DayNum"
    }
)
@dlt.expect_or_drop("valid_trip", "trip_count > 0")
@dlt.expect_or_drop("valid_duration", "duration_seconds > 0")
def silver_ev_trips():
    """Aggregate telemetry data into trip-level records.
    
    Transformation logic:
    1. Group by (DayNum, VehId, Trip) - each combination is one trip
    2. Calculate distance using Haversine formula from GPS coordinates
    3. Calculate total energy consumption
    4. Calculate trip duration
    5. Extract origin/destination coordinates
    6. Calculate average metrics (speed, battery SOC, elevation)
    
    Returns: One row per trip with calculated metrics
    """
    
    # Read bronze telemetry data (batch read for aggregation)
    df = spark.read.table(f"{catalog}.bronze.bronze_ev_telemetry")
    
    # Aggregate by trip (DayNum, VehId, Trip)
    trips = (
        df
        .groupBy("DayNum", "VehId", "Trip")
        .agg(
            # Temporal features
            spark_min("`Timestamp(ms)`").alias("start_timestamp_ms"),
            spark_max("`Timestamp(ms)`").alias("end_timestamp_ms"),
            
            # GPS coordinates for distance calculation
            first("`Latitude[deg]`").alias("origin_lat"),
            first("`Longitude[deg]`").alias("origin_lon"),
            last("`Latitude[deg]`").alias("destination_lat"),
            last("`Longitude[deg]`").alias("destination_lon"),
            
            # Energy metrics - handle potential string columns
            spark_sum(col("`Energy_Consumption`").cast("double")).alias("total_energy_consumption"),
            
            # Battery metrics - cast strings to double
            avg(col("`HV Battery SOC[%]`").cast("double")).alias("avg_battery_soc"),
            spark_min(col("`HV Battery SOC[%]`").cast("double")).alias("min_battery_soc"),
            spark_max(col("`HV Battery SOC[%]`").cast("double")).alias("max_battery_soc"),
            avg(col("`HV Battery Voltage[V]`").cast("double")).alias("avg_battery_voltage"),
            
            # Speed metrics - cast strings to double
            avg(col("`Vehicle Speed[km/h]`").cast("double")).alias("avg_speed_kmh"),
            spark_max(col("`Vehicle Speed[km/h]`").cast("double")).alias("max_speed_kmh"),
            
            # Elevation metrics
            avg("`Elevation Smoothed[m]`").alias("avg_elevation_m"),
            spark_max("`Elevation Smoothed[m]`").alias("max_elevation_m"),
            spark_min("`Elevation Smoothed[m]`").alias("min_elevation_m"),
            
            # Count of telemetry points
            count("*").alias("trip_count")
        )
    )
    
    # Calculate derived metrics
    result = (
        trips
        # Convert timestamp to datetime
        .withColumn("trip_start_datetime", 
                   from_unixtime(col("start_timestamp_ms") / 1000))
        .withColumn("trip_end_datetime", 
                   from_unixtime(col("end_timestamp_ms") / 1000))
        
        # Calculate duration in seconds and minutes
        .withColumn("duration_seconds", 
                   (col("end_timestamp_ms") - col("start_timestamp_ms")) / 1000)
        .withColumn("duration_minutes", 
                   spark_round(col("duration_seconds") / 60, 2))
        
        # Calculate distance using Haversine formula
        .withColumn("lat1_rad", radians(col("origin_lat")))
        .withColumn("lat2_rad", radians(col("destination_lat")))
        .withColumn("dlat", radians(col("destination_lat") - col("origin_lat")))
        .withColumn("dlon", radians(col("destination_lon") - col("origin_lon")))
        .withColumn("a", 
                   sin(col("dlat") / 2) * sin(col("dlat") / 2) +
                   cos(col("lat1_rad")) * cos(col("lat2_rad")) *
                   sin(col("dlon") / 2) * sin(col("dlon") / 2))
        .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a"))))
        .withColumn("distance_km", spark_round(6371 * col("c"), 2))
        
        # Calculate elevation gain
        .withColumn("elevation_gain_m", 
                   spark_round(col("max_elevation_m") - col("min_elevation_m"), 2))
        
        # Calculate efficiency (km per unit of energy)
        .withColumn("efficiency_km_per_energy",
                   spark_round(col("distance_km") / col("total_energy_consumption"), 4))
        
        # Temporal features
        .withColumn("trip_date", to_timestamp(col("trip_start_datetime")).cast("date"))
        .withColumn("trip_year", year(col("trip_start_datetime")))
        .withColumn("trip_month", month(col("trip_start_datetime")))
        .withColumn("trip_dayofweek", dayofweek(col("trip_start_datetime")))
        .withColumn("trip_hour", hour(col("trip_start_datetime")))
        
        # Round numeric columns
        .withColumn("total_energy_consumption", spark_round(col("total_energy_consumption"), 2))
        .withColumn("avg_battery_soc", spark_round(col("avg_battery_soc"), 2))
        .withColumn("avg_battery_voltage", spark_round(col("avg_battery_voltage"), 2))
        .withColumn("avg_speed_kmh", spark_round(col("avg_speed_kmh"), 2))
        .withColumn("max_speed_kmh", spark_round(col("max_speed_kmh"), 2))
        .withColumn("avg_elevation_m", spark_round(col("avg_elevation_m"), 2))
        
        # Add processing timestamp
        .withColumn("processing_timestamp", current_timestamp())
        
        # Drop intermediate calculation columns
        .drop("lat1_rad", "lat2_rad", "dlat", "dlon", "a", "c")
        
        # Select final columns in logical order
        .select(
            # Trip identifiers
            "DayNum", "VehId", "Trip",
            
            # Temporal features
            "trip_date", "trip_start_datetime", "trip_end_datetime",
            "start_timestamp_ms", "end_timestamp_ms",
            "trip_year", "trip_month", "trip_dayofweek", "trip_hour",
            
            # Trip metrics
            "distance_km", "duration_seconds", "duration_minutes",
            "total_energy_consumption", "efficiency_km_per_energy",
            
            # Location
            "origin_lat", "origin_lon", "destination_lat", "destination_lon",
            
            # Speed metrics
            "avg_speed_kmh", "max_speed_kmh",
            
            # Battery metrics
            "avg_battery_soc", "min_battery_soc", "max_battery_soc",
            "avg_battery_voltage",
            
            # Elevation metrics
            "avg_elevation_m", "max_elevation_m", "min_elevation_m",
            "elevation_gain_m",
            
            # Quality metrics
            "trip_count",
            
            # Metadata
            "processing_timestamp"
        )
    )
    
    return result
