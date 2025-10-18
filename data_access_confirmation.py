from pyspark.sql import SparkSession
import sys
import socket
#from pyspark.sql.functions import udf
import os
import logging
from pyspark import SparkFiles

# Viz
import matplotlib.pyplot as plt
import seaborn as sns

# Geospatial
import folium
from folium.plugins import HeatMap
import json
import warnings
from pathlib import Path
import numpy as np

def get_dataframes_and_logger():
    # Create SparkSession only when this function is called (on driver)
    spark = SparkSession.builder.appName("ExtractColumns").master("spark://spark-master:7077").getOrCreate()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("App")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("AppL - %(asctime)s - %(levelname)s - %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info("Starting the spark session")

    archive_dir = SparkFiles.get("environment")
    python_lib = os.path.join(archive_dir, "lib/python3.11/site-packages")
    sys.path.insert(0, python_lib)

#    def hostname():
 #       return socket.gethostname()

  #  host_udf = udf(hostname)

    # Load datasets
    crashes_df = spark.read.csv("file:///data/Traffic_Crashes_-_Crashes_20250901.csv", header=True)
   # df_with_host = crashes_df.withColumn("executor", host_udf())
   # df_with_host.groupBy("executor").count().show()
  # crahses_df.write.mode("overwrite").parquet("/app/data/crashes_clean.parquet")
#    for column in crashes_df.columns:
#        logger.info(f"column name :{column}")

#    logger.info(f"Crashes dataset shape: ({crashes_df.count()}, {len(crashes_df.columns)})")

    people_df = spark.read.csv("file:///data/Traffic_Crashes_-_People_20250901.csv", header=True)
#    for column in people_df.columns:
#        logger.info(f"column name :{column}")
#    logger.info(f"People dataset shape: ({people_df.count()}, {len(people_df.columns)})")

    vehicle_df = spark.read.csv("file:///data/Traffic_Crashes_-_Vehicles_20250901.csv", header=True)
#    for column in vehicle_df.columns:
#        logger.info(f"column name :{column}")
#    logger.info(f"Vehicle dataset shape: ({vehicle_df.count()}, {len(vehicle_df.columns)})")

    fatalities_df = spark.read.csv("file:///data/Traffic_Crashes_-_Vision_Zero_Chicago_Traffic_Fatalities_20250901.csv", header=True)
#    logger.info(f"Fatalities dataset shape: ({fatalities_df.count()}, {len(fatalities_df.columns)})")

    # Return all dataframes and logger for use by caller
    return crashes_df, people_df, vehicle_df, fatalities_df, logger

