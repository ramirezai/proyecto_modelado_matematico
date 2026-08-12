import dlt
from pyspark.sql.functions import col, current_timestamp

# Get configuration from pipeline settings
source_volume_path = spark.conf.get("source_volume_path")
catalog = spark.conf.get("catalog")

@dlt.table(
    name=f"{catalog}.bronze.bronze_ev_telemetry",
    comment="Raw EV telemetry data ingested from eVED dataset using AutoLoader. Contains second-by-second sensor data: GPS, battery, speed, energy consumption.",
    table_properties={
        "quality": "bronze",
        "pipelines.autoOptimize.zOrderCols": "VehId,DayNum",
        "delta.columnMapping.mode": "name"  # Enable column mapping for special characters
    }
)
def bronze_ev_telemetry():
    """Ingest raw telemetry data from eVED CSV files.
    
    Source: 54 CSV files (~490k rows each) with 35 columns of telemetry data
    Key columns:
    - DayNum, VehId, Trip: Trip identifiers
    - Timestamp(ms): Measurement timestamp
    - Latitude[deg], Longitude[deg]: GPS coordinates
    - Vehicle Speed[km/h]: Speed
    - HV Battery Current[A], HV Battery SOC[%], HV Battery Voltage[V]: Battery metrics
    - Energy_Consumption: Energy consumed
    - Elevation, Gradient: Terrain features
    
    AutoLoader automatically:
    - Detects schema from CSV files (35 columns)
    - Tracks processed files incrementally
    - Handles schema evolution
    
    Note: Column mapping is enabled to support special characters in column names
    """
    return (
        spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "csv")
            .option("cloudFiles.schemaLocation", f"/Volumes/{catalog}/bronze/eved/_schemas")
            .option("header", "true")
            .option("inferSchema", "true")
            .load(source_volume_path)
            .withColumn("ingestion_timestamp", current_timestamp())
    )
