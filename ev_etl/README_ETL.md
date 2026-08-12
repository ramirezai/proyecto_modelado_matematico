# EV Trip ETL Pipeline - Predicción de Consumo Energético

## Descripción

Pipeline ETL para procesamiento de datos de viajes de vehículos eléctricos (EV), diseñado para generar features de ML que permitan predecir el consumo energético en rutas específicas.

## Arquitectura

El pipeline sigue una arquitectura de medallón con 4 capas:

```
Volume (/Volumes/{catalog}/bronze/eved)
  ↓ [AutoLoader]
🥉 BRONZE → bronze_ev_trips
  ↓ [Limpieza y validación]
🥈 SILVER → silver_ev_trips  
  ↓ [Agregaciones]
🥇 GOLD → gold_trip_statistics, gold_route_analysis
  ↓ [Feature Engineering]
🤖 ML_FEATURES → ml_features_ev_energy_prediction
```

## Estructura del Proyecto

```
ev_etl/
├── databricks.yml                          # Configuración principal del bundle
├── resources/
│   └── ev_etl_pipeline.pipeline.yml        # Definición del pipeline SDP
└── src/
    └── pipelines/
        └── ev_etl/
            └── transformations/
                ├── 01_bronze.py            # Ingesta con AutoLoader
                ├── 02_silver.py            # Limpieza y validaciones
                ├── 03_gold.py              # Agregaciones y análisis
                └── 04_ml_features.py       # Features para ML
```

## Tablas Generadas

### 🥉 Bronze Layer
- **bronze_ev_trips**: Datos raw ingestados desde el volumen con AutoLoader

### 🥈 Silver Layer  
- **silver_ev_trips**: Datos limpios y validados con:
  - Checks de calidad (distancia, energía, duración > 0)
  - Features temporales (año, mes, día, hora)
  - Métricas calculadas (eficiencia km/kWh, velocidad promedio)

### 🥇 Gold Layer
- **gold_trip_statistics**: Estadísticas agregadas por vehículo y fecha
- **gold_route_analysis**: Análisis de rutas con patrones de consumo

### 🤖 ML Features Layer
- **ml_features_ev_energy_prediction**: Features para predicción de consumo energético
  - Features históricas del vehículo (últimos 30 viajes)
  - Features históricas de la ruta (últimos 10 viajes)
  - Features temporales
  - Target: `target_energy_kwh`

## Ambientes (Targets)

### Dev (Desarrollo)
- Catalog: `dev`
- Esquemas: `dev.bronze`, `dev.silver`, `dev.gold`, `dev.ml_features`
- Volume source: `/Volumes/dev/bronze/eved`
- Mode: development

### Prod (Producción)
- Catalog: `prod`
- Esquemas: `prod.bronze`, `prod.silver`, `prod.gold`, `prod.ml_features`
- Volume source: `/Volumes/prod/bronze/eved`
- Mode: production
- Runs as: joel@ramirezai.com

## Comandos Principales

### Validar configuración
```bash
databricks bundle validate --strict --target dev
databricks bundle validate --strict --target prod
```

### Desplegar el pipeline
```bash
# Desplegar en dev
databricks bundle deploy --target dev

# Desplegar en prod
databricks bundle deploy --target prod
```

### Ejecutar el pipeline
```bash
# Ejecutar en dev
databricks bundle run ev_etl --target dev

# Ejecutar en prod
databricks bundle run ev_etl --target prod
```

### Ver resumen del bundle
```bash
databricks bundle summary --target dev
```

## Requisitos de Datos

### Formato de entrada (CSV en volumen)
El pipeline espera archivos CSV en `/Volumes/{catalog}/bronze/eved` con las siguientes columnas mínimas:

- `trip_id`: Identificador único del viaje
- `vehicle_id`: Identificador del vehículo
- `trip_date`: Fecha del viaje
- `trip_timestamp`: Timestamp del inicio del viaje
- `origin`: Origen del viaje
- `destination`: Destino del viaje
- `distance_km`: Distancia recorrida en kilómetros
- `energy_kwh`: Energía consumida en kWh
- `duration_minutes`: Duración del viaje en minutos

### Schema Evolution
AutoLoader detecta automáticamente cambios en el schema y los registra en:
`/Volumes/{catalog}/bronze/eved/_schemas`

## Features para ML

La tabla `ml_features_ev_energy_prediction` contiene:

**Target Variable:**
- `target_energy_kwh`: Energía consumida (variable a predecir)

**Features Principales:**
- `distance_km`: Distancia del viaje
- `duration_minutes`: Duración del viaje
- `avg_speed_kmh`: Velocidad promedio

**Features Temporales:**
- `trip_year`, `trip_month`, `trip_dayofweek`, `trip_hour`

**Features Históricas del Vehículo:**
- `vehicle_avg_efficiency_last30`: Eficiencia promedio (últimos 30 viajes)
- `vehicle_avg_energy_last30`: Consumo promedio (últimos 30 viajes)
- `vehicle_stddev_energy_last30`: Desviación estándar del consumo
- `vehicle_trip_count_last30`: Número de viajes recientes

**Features Históricas de Ruta:**
- `route_avg_energy_last10`: Consumo promedio en esta ruta
- `route_avg_duration_last10`: Duración promedio en esta ruta
- `route_trip_count`: Frecuencia de la ruta

## Checks de Calidad

El pipeline implementa data quality checks en la capa Silver:

- ✅ `valid_distance`: Distancia debe ser > 0
- ✅ `valid_energy`: Energía debe ser > 0  
- ✅ `valid_duration`: Duración debe ser > 0
- ✅ Filtrado de valores nulos en columnas críticas

Registros que no cumplen estos checks son descartados automáticamente.

## Monitoreo

Después de ejecutar el pipeline:

1. **Ver en la UI**: 
   - Ve a "Workflows" → "Lakeflow Spark Declarative Pipelines"
   - Busca "EV Trip ETL Pipeline - {catalog}"
   - Revisa el lineage y las métricas de calidad

2. **Consultar métricas**:
```sql
-- Verificar datos en bronze
SELECT COUNT(*) FROM dev.bronze.bronze_ev_trips;

-- Verificar datos en silver
SELECT COUNT(*) FROM dev.silver.silver_ev_trips;

-- Explorar features de ML
SELECT * FROM dev.ml_features.ml_features_ev_energy_prediction LIMIT 10;
```

## Próximos Pasos

1. **Entrenamiento del Modelo**:
   - Usar tabla `ml_features_ev_energy_prediction`
   - Target: `target_energy_kwh`
   - Considerar modelos: XGBoost, Random Forest, LightGBM

2. **Optimizaciones**:
   - Ajustar ventanas de features históricas según disponibilidad de datos
   - Agregar features de clima/tráfico si están disponibles
   - Implementar feature store para reutilización

3. **Monitoreo**:
   - Configurar alertas sobre data quality metrics
   - Dashboards de consumo energético
   - Tracking de drift en el modelo

## Troubleshooting

### Error: "Table not found"
- Asegúrate de que los catálogos `dev`/`prod` existen en Unity Catalog
- Verifica que tienes permisos para crear esquemas y tablas

### Error: "Volume not found"  
- Verifica que el volumen existe: `/Volumes/{catalog}/bronze/eved`
- Asegúrate de tener permisos de lectura en el volumen

### Pipeline falla en validación de calidad
- Revisa los datos de entrada en el volumen
- Verifica que no hay valores negativos o nulos en columnas críticas
- Consulta las métricas del pipeline en la UI

## Contacto

Proyecto: Modelado Matemático - Predicción de Consumo EV  
Owner: joel@ramirezai.com
