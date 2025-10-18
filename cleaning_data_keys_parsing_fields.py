import sys
#sys.path.append('./app')
print("i am printing system path below")
#print(sys.path)
import pandas as pd
import data_access_confirmation
print("i reached here")
from data_access_confirmation import get_dataframes_and_logger

import seaborn as sns
import folium, json
from folium.plugins import HeatMap
import warnings
import matplotlib.pyplot as plt
import numpy as np

from pyspark.sql.functions import to_timestamp, col
#from pyspark.sql import SparkSession
#from pyspark.sql.functions import col
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import when, avg, year, month, dayofweek, hour,   upper, trim, nanvl,  count, sum,  lit
from pyspark.sql.window import Window
from pyspark.sql.functions import to_date, date_trunc, count as spark_count

TILE_URL = 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png'

crashes_df, people_df, vehicle_df, fatalities_df, logger = get_dataframes_and_logger()


print("number of columns in crashes", len(crashes_df.columns))
print("number of columns in crashes", len(crashes_df.columns)) 
          


# Capitalizing the column names of each dataframe for consistency
crashes_df = crashes_df.select([col(c).alias(c.upper()) for c in crashes_df.columns])
people_df = people_df.select([col(c).alias(c.upper()) for c in people_df.columns])
vehicle_df = vehicle_df.select([col(c).alias(c.upper()) for c in vehicle_df.columns])
fatalities_df = fatalities_df.select([col(c).alias(c.upper()) for c in fatalities_df.columns])

# key ID in Crashes dataframe
CRASH_ID = 'CRASH_RECORD_ID' if 'CRASH_RECORD_ID' in crashes_df.columns else crashes_df.columns[0]

# Parsing dates
def parse_crash_datetime(df, col_options):
    found = next((c for c in col_options if c in df.columns), None)
    if found:
        return df.withColumn('CRASH_DATETIME', to_timestamp(col(found)))
    return df

crashes_df = parse_crash_datetime(crashes_df, ['CRASH_DATE', 'CRASH_DATE_EST_I', 'CRASH_DATE_TIMESTAMP'])
fatalities_df = parse_crash_datetime(fatalities_df, ['CRASH_DATE'])

# Getting YEAR, MONTH, DAY OF WEEK AND HOUR from Crashes dataframe
if 'CRASH_DATETIME' in crashes_df.columns:
    crashes_df = crashes_df.withColumn('YEAR', year(col('CRASH_DATETIME'))) \
                     .withColumn('MONTH', month(col('CRASH_DATETIME'))) \
                     .withColumn('DAY_OF_WEEK', dayofweek(col('CRASH_DATETIME'))) \
                     .withColumn('HOUR', hour(col('CRASH_DATETIME')))

# Lat and Long coordinates
def add_lat_lon(df):
    lon_col = next((c for c in ['LONGITUDE','X_COORDINATE','LON'] if c in df.columns), None)
    lat_col = next((c for c in ['LATITUDE','Y_COORDINATE','LAT'] if c in df.columns), None)
    if lon_col:
        df = df.withColumn('LON', col(lon_col).cast(DoubleType()))
    if lat_col:
        df = df.withColumn('LAT', col(lat_col).cast(DoubleType()))
    return df

crashes_df = add_lat_lon(crashes_df)
fatalities_df = add_lat_lon(fatalities_df)
severity_cols = [c for c in crashes_df.columns if 'INJURIES' in c or 'FATAL' in c]
print(severity_cols[:10], CRASH_ID)


people_summary = None
if CRASH_ID in people_df.columns:
    # Use PySpark functions for conditional aggregation
    persons_agg = count("PERSON_ID").alias("persons")
    
    if 'PEDPED_FLAG' in people_df.columns:
        pedestrians_agg = sum("PEDPED_FLAG").alias("pedestrians")
    else:
        pedestrians_agg = sum(when(col("PERSON_TYPE") == "PEDESTRIAN", 1).otherwise(0)).alias("pedestrians")

    if 'PERSON_TYPE' in people_df.columns:
        cyclists_agg = sum(when(col("PERSON_TYPE") == "BICYCLE", 1).otherwise(0)).alias("cyclists")
    else:
        cyclists_agg = count("PERSON_ID").alias("cyclists")

    people_summary = people_df.groupBy(CRASH_ID).agg(
        persons_agg,
        pedestrians_agg,
        cyclists_agg
    )
else:
    people_summary = spark.createDataFrame([], people_df.schema)


vehicles_summary = None
if CRASH_ID in vehicle_df.columns:
    # Use PySpark functions for conditional aggregation
    vehicles_agg = count("VEHICLE_ID").alias("vehicles")
    
    if 'PRIM_CONTRIBUTORY_CAUSE' in vehicle_df.columns:
        speeding_agg = sum(
            when(col("PRIM_CONTRIBUTORY_CAUSE").contains("SPEED"), 1).otherwise(0)
        ).alias("speeding_veh")
    else:
        speeding_agg = count("VEHICLE_ID").alias("speeding_veh")

    vehicles_summary = vehicle_df.groupBy(CRASH_ID).agg(
        vehicles_agg,
        speeding_agg
    )
else:
    vehicles_summary = spark.createDataFrame([], vehicle_df.schema)


combined = crashes_df.alias("combined")

if people_summary and people_summary.count() > 0:
    combined = combined.join(
        people_summary.alias("people_sum"),
        on=[CRASH_ID],
        how="left"
    )

if vehicles_summary and vehicles_summary.count() > 0:
    combined = combined.join(
        vehicles_summary.alias("vehicles_sum"),
        on=[CRASH_ID],
        how="left"
    )

combined.show(n=1, vertical=True)
print((combined.count(), len(combined.columns)))
print("combined columns")
print(combined.columns)

for column_name in ['PRIM_CONTRIBUTORY_CAUSE', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'TRAFFIC_CONTROL_DEVICE']:
    if column_name in combined.columns:
        combined = combined.withColumn(
            column_name,
            when(
                # Check for 'NAN' or 'NONE' after converting to upper case
                upper(trim(col(column_name))).isin(['NAN', 'NONE']),
                None  # Replace with null (PySpark's representation of missing)
            ).otherwise(
                upper(trim(col(column_name)))  # Otherwise, keep the cleaned string
            )
        )

# Applying filters in Lat and Long coordinates
valid_geo_condition = (col("LAT").between(41.60, 42.10)) & (col("LON").between(-88.0, -87.3))

# Use `when` and `otherwise` to conditionally set LAT and LON to null
combined_clean = combined.withColumn(
    "LAT",
    when(~valid_geo_condition, None).otherwise(col("LAT"))
).withColumn(
    "LON",
    when(~valid_geo_condition, None).otherwise(col("LON"))
)

combined_clean.show(1, vertical=True)
print((combined_clean.count(), len(combined_clean.columns)))

from pyspark.sql.functions import expr, to_date, date_trunc, count as spark_count
import matplotlib.pyplot as plt


if 'CRASH_DATE' in combined_clean.columns:
    combined_clean = combined_clean.withColumn(
        'CRASH_DATETIME',
        expr("try_to_timestamp(CRASH_DATE, 'MM/dd/yyyy hh:mm:ss a')")
    )


if 'CRASH_DATETIME' in combined_clean.columns:
    filtered = combined_clean.filter(combined_clean['CRASH_DATETIME'].isNotNull())

  
    daily = (filtered
             .withColumn('CRASH_DATE', to_date('CRASH_DATETIME'))
             .groupBy('CRASH_DATE')
             .agg(spark_count('*').alias('count'))
             .orderBy('CRASH_DATE')
             .toPandas())
    
    monthly = (filtered
               .withColumn('CRASH_MONTH', date_trunc('month', 'CRASH_DATETIME'))
               .groupBy('CRASH_MONTH')
               .agg(spark_count('*').alias('count'))
               .orderBy('CRASH_MONTH')
               .toPandas())

    fig, axes = plt.subplots(2, 1, figsize=(12,8), sharex=False)
    daily['count'].rolling(7).mean().plot(ax=axes[0], color='tab:blue', title='Daily crashes (7-day rolling)')
    monthly['count'].plot(ax=axes[1], color='tab:orange', title='Monthly crashes')
    plt.tight_layout()
    plt.savefig("Daily_Monthly_Crashes.png")
else:
    print('CRASH_DATETIME missing; skipping time series.')
 

if 'CRASH_DATE' in combined_clean.columns:
    combined_clean = combined_clean.withColumn(
        'CRASH_DATETIME',
        expr("try_to_timestamp(CRASH_DATE, 'MM/dd/yyyy hh:mm:ss a')")
    )

if 'CRASH_DATETIME' in combined_clean.columns:
    filtered = combined_clean.filter(combined_clean['CRASH_DATETIME'].isNotNull())
    filtered = filtered.withColumn('HOUR', hour(filtered['CRASH_DATETIME']))
    filtered = filtered.withColumn('DAY_OF_WEEK_NUM', dayofweek(filtered['CRASH_DATETIME']))
    day_map = {1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday', 5: 'Thursday', 6: 'Friday', 7: 'Saturday'}
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType
    day_name_udf = udf(lambda x: day_map.get(x, 'Unknown'), StringType())
    filtered = filtered.withColumn('DAY_OF_WEEK', day_name_udf(filtered['DAY_OF_WEEK_NUM']))

    pdf = filtered.select('HOUR', 'DAY_OF_WEEK').dropna().toPandas()

    fig, axes = plt.subplots(1, 2, figsize=(14,5))
    pdf['HOUR'] = pdf['HOUR'].astype(int)
    pdf['DAY_OF_WEEK'] = pdf['DAY_OF_WEEK'].astype(str)
    sns.countplot(data=pdf, x='HOUR', ax=axes[0], color='tab:purple')
    axes[0].set_title('Crashes by hour of day')
    axes[0].set_xlabel('Hour')

    dow_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
    sns.countplot(data=pdf, x='DAY_OF_WEEK', order=dow_order, ax=axes[1], color='tab:green')
    axes[1].set_title('Crashes by day of week')
    axes[1].tick_params(axis='x', rotation=30)
    plt.tight_layout()
    plt.savefig("crash_by_hour_and_day_week.png")
else:
    print('Time parts missing; skipping HOUR/DOW charts.')


from pyspark.sql.functions import col, when, count, expr
import matplotlib.pyplot as plt
import seaborn as sns

sev_cols = [c for c in combined_clean.columns if c.startswith('INJURIES') or c.endswith('FATAL')]

summary = {}
if sev_cols:
    expr_str = " OR ".join([f"({c} > 0)" for c in sev_cols])
    filtered = combined_clean.withColumn('injury_any', expr(expr_str))
    summary['injury_any'] = filtered.filter(col('injury_any')).count() / filtered.count() if filtered.count() > 0 else 0
else:
    filtered = combined_clean

if 'PRIM_CONTRIBUTORY_CAUSE' in filtered.columns:
    top_causes_df = (filtered.groupBy('PRIM_CONTRIBUTORY_CAUSE')
                     .count()
                     .orderBy(col('count').desc())
                     .limit(12)
                     .toPandas())
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_causes_df['count'], y=top_causes_df['PRIM_CONTRIBUTORY_CAUSE'], orient='h', color='tab:red')
    plt.title('Top primary contributory causes')
    plt.xlabel('Count')
    plt.tight_layout()
    plt.savefig("primary_causes.png")

if sev_cols and 'WEATHER_CONDITION' in filtered.columns:
    weather_df = (filtered.groupBy('WEATHER_CONDITION')
                  .agg((count(when(col('injury_any'), True)) / count('*')).alias('injury_rate'))
                  .orderBy(col('injury_rate').desc())
                  .limit(10)
                  .toPandas())
    plt.figure(figsize=(10,4))
    sns.barplot(x=weather_df['injury_rate'], y=weather_df['WEATHER_CONDITION'], orient='h', color='tab:blue')
    plt.title('Highest injury rates by weather (top 10)')
    plt.xlabel('Injury rate')
    plt.tight_layout()
    plt.savefig("highest_injury.png")


import folium
from folium.plugins import HeatMap

if 'LAT' in combined_clean.columns and 'LON' in combined_clean.columns:
    coords_df = combined_clean.select('LAT', 'LON').dropna()
    coords_pd = coords_df.toPandas()
    if not coords_pd.empty:
        center = [coords_pd['LAT'].median(), coords_pd['LON'].median()]
        m = folium.Map(location=center, zoom_start=11, tiles=TILE_URL, attr='OSM')
        hm_data = coords_pd.sample(min(50000, len(coords_pd))).values.tolist()
        HeatMap(hm_data, radius=9, blur=12, max_zoom=13).add_to(m)
        map_path = 'crash_heatmap_pyspark.html'
        m.save(str(map_path))
        print(f'Heatmap saved to {map_path}')
    else:
        print('No valid coordinates for mapping.')
else:
    print('No valid coordinates for mapping.')



from folium.plugins import MarkerCluster


if 'LAT' in combined_clean.columns and 'LON' in combined_clean.columns:
    coords_df = combined_clean.select('LAT', 'LON', CRASH_ID, 'CRASH_DATETIME').dropna()
    coords_pd = coords_df.toPandas()
    if not coords_pd.empty:
        center = [coords_pd['LAT'].median(), coords_pd['LON'].median()]
        m2 = folium.Map(location=center, zoom_start=11, tiles=TILE_URL, attr='OSM')
        cluster = MarkerCluster()

        subset = coords_pd.head(3000)
        for _, row in subset.iterrows():
            popup = f"{row.get(CRASH_ID, '')} | {row.get('CRASH_DATETIME','')}"
            folium.Marker([row['LAT'], row['LON']], popup=popup).add_to(cluster)

        cluster.add_to(m2)
        map_path2 = 'crash_markers_pyspark.html'
        m2.save(str(map_path2))
        print(f'Clustered markers map saved to {map_path2}')
    else:
        print('No valid coordinates for mapping.')
else:
    print('No valid coordinates for mapping.')


from pyspark.sql.functions import col, when, expr, round as spark_round

cc = combined_clean

sev_cols = [c for c in cc.columns if c.startswith('INJURIES') or c.endswith('FATAL')]
if sev_cols:
    expr_str = " OR ".join([f"({c} > 0)" for c in sev_cols])
    cc = cc.withColumn('injury_any', expr(expr_str))
else:
    cc = cc.withColumn('injury_any', expr("false"))

fatal_col = next((c for c in cc.columns if 'FATAL' in c), None)
if fatal_col is not None:
    cc = cc.withColumn('is_fatal', when(col(fatal_col).cast('double') > 0, True).otherwise(False))
else:
    if CRASH_ID in fatal.columns and CRASH_ID in cc.columns:
        fatal_ids = set(fatal.select(CRASH_ID).dropna().toPandas()[CRASH_ID].astype(str))
        cc = cc.withColumn('is_fatal', col(CRASH_ID).cast('string').isin(list(fatal_ids)))
    else:
        cc = cc.withColumn('is_fatal', expr("false"))

cc = cc.filter(col('LAT').isNotNull() & col('LON').isNotNull())

if 'HOUR' in cc.columns:
    cc = cc.withColumn('period', when((col('HOUR') >= 6) & (col('HOUR') <= 19), 'DAY').otherwise('NIGHT'))
else:
    cc = cc.withColumn('period', when((hour(col('CRASH_DATETIME')) >= 6) & (hour(col('CRASH_DATETIME')) <= 19), 'DAY').otherwise('NIGHT'))


cc = cc.withColumn('lat_bin', spark_round(col('LAT'), 3)).withColumn('lon_bin', spark_round(col('LON'), 3))


agg_df = cc.groupBy('lat_bin', 'lon_bin').agg(
    count('LAT').alias('count'),
    count(when(col('injury_any'), True)).alias('injury_any_count'),
    count(when(col('is_fatal'), True)).alias('fatal_count')
)
agg_df = agg_df.withColumn('lat', col('lat_bin')).withColumn('lon', col('lon_bin'))


cause_col = 'PRIM_CONTRIBUTORY_CAUSE' if 'PRIM_CONTRIBUTORY_CAUSE' in cc.columns else None
if cause_col:
    from pyspark.sql import Window
    from pyspark.sql.functions import row_number

    
    cause_counts = cc.groupBy('lat_bin', 'lon_bin', cause_col).count()
    w = Window.partitionBy('lat_bin', 'lon_bin').orderBy(col('count').desc())
    cause_counts = cause_counts.withColumn('rn', row_number().over(w))
    top_cause_df = cause_counts.filter(col('rn') == 1).select('lat_bin', 'lon_bin', cause_col)
    top_cause_df = top_cause_df.withColumnRenamed(cause_col, 'top_cause')
    agg_df = agg_df.join(top_cause_df, ['lat_bin', 'lon_bin'], 'left')
else:
    agg_df = agg_df.withColumn('top_cause', expr("''"))


from pyspark.sql.functions import when as spark_when
agg_df = agg_df.withColumn('injury_rate', spark_when(col('count') > 0, col('injury_any_count') / col('count')).otherwise(0))
agg_df = agg_df.withColumn('fatal_flag', col('fatal_count') > 0)
agg_pd = agg_df.orderBy(col('count').desc(), col('fatal_count').desc(), col('injury_rate').desc()).toPandas()
hotspots_top = agg_pd.head(200).copy()
print(hotspots_top.shape)


from pyspark.sql.functions import expr, col
import folium
from folium.plugins import HeatMap
from folium import FeatureGroup, CircleMarker
import pandas as pd


heat_limit = 50000
if hasattr(cc, "toPandas"):
    for date_col in ['CRASH_DATE', 'CRASH_DATE_EST_I', 'CRASH_DATE_TIMESTAMP', 'DATE_POLICE_NOTIFIED']:
        if date_col in cc.columns and 'CRASH_DATETIME' not in cc.columns:
            cc = cc.withColumn('CRASH_DATETIME', expr(f"try_to_timestamp({date_col}, 'MM/dd/yyyy hh:mm:ss a')"))

    if 'LAT' in cc.columns and 'LON' in cc.columns:
        cc_valid = cc.filter(col('LAT').isNotNull() & col('LON').isNotNull())
    else:
        cc_valid = None

    if cc_valid is not None:
        select_cols = [col('LAT').cast('double'), col('LON').cast('double')]
        if 'period' in cc_valid.columns:
            select_cols.append(col('period'))
        if 'is_fatal' in cc_valid.columns:
            select_cols.append(col('is_fatal'))
        if 'CRASH_DATETIME' in cc_valid.columns:
            select_cols.append(col('CRASH_DATETIME').cast('string').alias('CRASH_DATETIME'))
        if 'PRIM_CONTRIBUTORY_CAUSE' in cc_valid.columns:
            select_cols.append(col('PRIM_CONTRIBUTORY_CAUSE'))

        try:
            cc_pd = cc_valid.select(*select_cols).limit(heat_limit).toPandas()
        except Exception:
            cc_pd = cc_valid.select(col('LAT').cast('double').alias('LAT'),
                                     col('LON').cast('double').alias('LON')).limit(heat_limit).toPandas()
    else:
        cc_pd = pd.DataFrame(columns=['LAT','LON'])
else:
    cc_pd = cc.copy()

hotspots_pd = hotspots_top.toPandas() if hasattr(hotspots_top, "toPandas") else hotspots_top

if not cc_pd.empty:
    center = [cc_pd['LAT'].median(), cc_pd['LON'].median()]
    m = folium.Map(location=center, zoom_start=11, tiles=TILE_URL, attr='OSM')

    layer_hotspots = FeatureGroup(name='Hotspots (top 200)')
    if hotspots_pd is not None and len(hotspots_pd):
        for _, r in hotspots_pd.iterrows():
            lat = r.get('lat') or r.get('LAT')
            lon = r.get('lon') or r.get('LON')
            if pd.notna(lat) and pd.notna(lon):
                radius = max(4, min(30, int((r.get('count', 1) or 1) ** 0.5 * 3)))
                color = '#d73027' if (r.get('fatal_count', 0) or 0) > 0 else '#1a9850'
                popup = (f"Count: {int(r.get('count',0))}<br>Injuries: {int(r.get('injury_any_count',0))}"
                         f"<br>Fatal: {int(r.get('fatal_count',0))}<br>Top cause: {r.get('top_cause','')}")
                CircleMarker(location=[float(lat), float(lon)], radius=radius,
                             color=color, fill=True, fill_opacity=0.6, popup=popup).add_to(layer_hotspots)
    layer_hotspots.add_to(m)

    if 'period' in cc_pd.columns:
        day_coords = cc_pd.loc[cc_pd['period']=='DAY', ['LAT','LON']].dropna().values.tolist()
        if day_coords:
            HeatMap(day_coords[:heat_limit], name='Day Heatmap', radius=10, blur=12, max_zoom=13).add_to(m)
        night_coords = cc_pd.loc[cc_pd['period']=='NIGHT', ['LAT','LON']].dropna().values.tolist()
        if night_coords:
            HeatMap(night_coords[:heat_limit], name='Night Heatmap', radius=10, blur=12, max_zoom=13).add_to(m)
    else:
        coords = cc_pd[['LAT','LON']].dropna()
        if not coords.empty:
            HeatMap(coords.values.tolist()[:heat_limit], radius=10, blur=12, max_zoom=13).add_to(m)

    if 'is_fatal' in cc_pd.columns:
        fat = cc_pd[cc_pd['is_fatal'] == True].dropna(subset=['LAT','LON']).head(5000)
        if not fat.empty:
            fatal_points = FeatureGroup(name='Fatalities')
            for _, row in fat.iterrows():
                popup = f"Fatal crash | {row.get('CRASH_DATETIME','')} | Cause: {row.get('PRIM_CONTRIBUTORY_CAUSE','')}"
                CircleMarker([row['LAT'], row['LON']], radius=4, color='#000', fill=True, fill_opacity=0.9, popup=popup).add_to(fatal_points)
            fatal_points.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    admin_map_path = 'crash_admin_map_pyspark.html'
    m.save(str(admin_map_path))
    print(f'Admin map saved to {admin_map_path}')
else:
    print('No data for admin map')




from pyspark.sql.functions import col, upper, trim, expr, sum as spark_sum, count as spark_count


policy = combined_clean
cause_col = 'PRIM_CONTRIBUTORY_CAUSE' if 'PRIM_CONTRIBUTORY_CAUSE' in policy.columns else None
if cause_col:
    policy = policy.withColumn(cause_col, upper(trim(col(cause_col))))
else:
    policy = policy.withColumn('PRIM_CONTRIBUTORY_CAUSE', expr("'UNKNOWN'"))
    cause_col = 'PRIM_CONTRIBUTORY_CAUSE'

col_fatal = next((c for c in policy.columns if c.upper() == 'INJURIES_FATAL'), None)
col_sev = next((c for c in policy.columns if c.upper() == 'INJURIES_INCAPACITATING'), None)
col_nonsev = next((c for c in policy.columns if c.upper() == 'INJURIES_NON_INCAPACITATING'), None)
col_possible = next((c for c in policy.columns if c.upper() == 'INJURIES_POSSIBLE'), None)


policy = policy.withColumn('inj_fatal', col(col_fatal).cast('double')) if col_fatal else policy.withColumn('inj_fatal', expr("0"))
policy = policy.withColumn('inj_sev', col(col_sev).cast('double')) if col_sev else policy.withColumn('inj_sev', expr("0"))
policy = policy.withColumn('inj_nonsev', col(col_nonsev).cast('double')) if col_nonsev else policy.withColumn('inj_nonsev', expr("0"))
policy = policy.withColumn('inj_possible', col(col_possible).cast('double')) if col_possible else policy.withColumn('inj_possible', expr("0"))

policy = policy.withColumn(
    'impact_score',
    col('inj_fatal') * 10 +
    col('inj_sev') * 5 +
    col('inj_nonsev') * 2 +
    col('inj_possible') * 1
)

cause_agg = (policy.groupBy(cause_col)
             .agg(
                 spark_count('LAT').alias('crashes'),
                 spark_sum('impact_score').alias('total_impact'),
                 spark_sum('inj_fatal').alias('fatalities'),
                 spark_sum('inj_sev').alias('severe_injuries'),
                 spark_sum('inj_nonsev').alias('nonsevere_injuries'),
                 spark_sum('inj_possible').alias('possible_injuries')
             )
             .orderBy(col('total_impact').desc(), col('fatalities').desc(), col('crashes').desc())
            )

top5_causes = cause_agg.limit(5).toPandas().rename(columns={cause_col:'cause'})
print('Top-5 causes by impact:')
print(top5_causes)

from pyspark.sql.functions import expr, col
import folium
from folium.plugins import HeatMap
from folium import FeatureGroup, CircleMarker
import pandas as pd
import numpy as np
from pyspark.sql.functions import expr as _expr, col as _col

def safe_to_pandas(spark_df, cols, limit=50000):
	if 'CRASH_DATETIME' not in spark_df.columns:
		for date_col in ['CRASH_DATE', 'CRASH_DATE_EST_I', 'CRASH_DATE_TIMESTAMP', 'DATE_POLICE_NOTIFIED']:
			if date_col in spark_df.columns:
				spark_df = spark_df.withColumn('CRASH_DATETIME', _expr(f"try_to_timestamp({date_col}, 'MM/dd/yyyy hh:mm:ss a')"))
				break
	
	projections = []
	for c in cols:
		if isinstance(c, str):
			if c == 'CRASH_DATETIME':
				projections.append(_col(c).cast('string').alias('CRASH_DATETIME'))
			else:
				projections.append(_col(c))
		else:
			projections.append(c)
	
	try:
		pd_df = spark_df.select(*projections).limit(int(limit)).toPandas()
	except Exception:
		try:
			pd_df = spark_df.select(_col('LAT').cast('double').alias('LAT'), _col('LON').cast('double').alias('LON')).limit(int(limit)).toPandas()
		except Exception:
			pd_df = pd.DataFrame(columns=[c if isinstance(c, str) else str(c) for c in cols])
	return pd_df


from folium import FeatureGroup
from folium.plugins import HeatMap


HEAT_LIMIT = 50000
CIRCLE_LIMIT = 2000


cause_col = 'PRIM_CONTRIBUTORY_CAUSE' if ('PRIM_CONTRIBUTORY_CAUSE' in (policy.columns if hasattr(policy, 'columns') else [])) else None


if hasattr(policy, "toPandas"):
    for src in ('CRASH_DATE','CRASH_DATE_EST_I','CRASH_DATE_TIMESTAMP','DATE_POLICE_NOTIFIED'):
        if src in policy.columns and 'CRASH_DATETIME' not in policy.columns:
            policy = policy.withColumn('CRASH_DATETIME', expr(f"try_to_timestamp({src}, 'MM/dd/yyyy hh:mm:ss a')"))
            break

    
    if 'LAT' in policy.columns and 'LON' in policy.columns:
        pol_valid = policy.filter(col('LAT').isNotNull() & col('LON').isNotNull())
    else:
        pol_valid = None

    if pol_valid is not None:
        proj = [col('LAT').cast('double').alias('LAT'), col('LON').cast('double').alias('LON')]
        for c in ('impact_score','inj_fatal','inj_sev','period','PRIM_CONTRIBUTORY_CAUSE'):
            if c in pol_valid.columns:
                proj.append(col(c))
        if 'CRASH_DATETIME' in pol_valid.columns:
            proj.append(col('CRASH_DATETIME').cast('string').alias('CRASH_DATETIME'))
        try:
            policy_pd = pol_valid.select(*proj).limit(HEAT_LIMIT).toPandas()
        except Exception:
            policy_pd = pol_valid.select(col('LAT').cast('double').alias('LAT'),
                                         col('LON').cast('double').alias('LON')).limit(HEAT_LIMIT).toPandas()
    else:
        policy_pd = pd.DataFrame(columns=['LAT','LON'])
else:
    policy_pd = policy.copy()


top5_pd = top5_causes.toPandas() if hasattr(top5_causes, "toPandas") else top5_causes


if not policy_pd.empty and top5_pd is not None and len(top5_pd):
    center = [policy_pd['LAT'].median(), policy_pd['LON'].median()]
    cause_map = folium.Map(location=center, zoom_start=11, tiles=TILE_URL, attr='OSM')

    palette = ['#d7191c', '#fdae61', '#2c7bb6', '#abd9e9', '#1a9641']
    for idx, row in top5_pd.iterrows():
        cause = row.get('cause') if 'cause' in row else (row.get(cause_col) if cause_col else None)
        if cause is None:
            continue
        layer = FeatureGroup(name=f"Cause: {cause}")

       
        if cause_col and cause_col in policy_pd.columns:
            mask = policy_pd[cause_col] == cause
        else:
            mask = slice(None)

        
        desired_cols = ['LAT','LON','impact_score','inj_fatal','inj_sev']
        cols_present = [c for c in desired_cols if c in policy_pd.columns]

        
        if 'LAT' not in cols_present or 'LON' not in cols_present:
            
            continue

        subset = policy_pd.loc[mask, cols_present].dropna()

        
        if not subset.empty:
            HeatMap(subset[['LAT','LON']].values.tolist()[:HEAT_LIMIT], radius=10, blur=12, max_zoom=13).add_to(layer)

        
        sort_keys = [c for c in ['inj_fatal','inj_sev','impact_score'] if c in subset.columns]
        if sort_keys:
            samp = subset.sort_values(sort_keys, ascending=False).head(CIRCLE_LIMIT)
        else:
            samp = subset.head(CIRCLE_LIMIT)

        for _, r in samp.iterrows():
            impact = float(r.get('impact_score') or 0)
            inj_fatal_val = r.get('inj_fatal') if 'inj_fatal' in r.index else 0
            color = palette[idx % len(palette)] if (inj_fatal_val == 0 or pd.isna(inj_fatal_val)) else '#000000'
            radius = 3 + min(6, int((impact) ** 0.5))
            folium.CircleMarker([float(r['LAT']), float(r['LON'])], radius=radius,
                                color=color, fill=True, fill_opacity=0.7).add_to(layer)
        layer.add_to(cause_map)

    folium.LayerControl(collapsed=False).add_to(cause_map)
    cause_map_path = 'crash_top5_causes_map_pyspark.html'
    cause_map.save(str(cause_map_path))
    print(f'Top-5 cause map saved to {cause_map_path}')
else:
    print('Insufficient data for top-5 cause map')



from pyspark.sql.functions import col, upper, trim, hour, year, expr
from pyspark.sql.types import DoubleType


people_df = people_df.select([col(c).alias(c.upper()) for c in people_df.columns])
vehicle_df = vehicle_df.select([col(c).alias(c.upper()) for c in vehicle_df.columns])
crashes_df = crashes_df.select([col(c).alias(c.upper()) for c in crashes_df.columns])

crashes, people, vehicle, fatal = crashes_df, people_df, vehicle_df, fatalities_df

vehicles = vehicle_df

if 'CRASH_DATE' in crashes.columns:
    crashes = crashes.withColumnRenamed('CRASH_DATE', 'CRASH_DATE_SOURCE')


CRASH_ID = 'CRASH_RECORD_ID' if 'CRASH_RECORD_ID' in crashes.columns else (crashes.columns[0] if len(crashes.columns)>0 else None)


if 'PERSON_TYPE' in people.columns:
    people = people.filter(upper(trim(col('PERSON_TYPE'))) == 'DRIVER')


state_col = next((c for c in people.columns if c in ['DRIVER_LICENSE_STATE','DRVR_LIC_STATE','LIC_PLATE_STATE']), None)
age_col = next((c for c in people.columns if c in ['AGE','DRIVER_AGE','PERSON_AGE']), None)


veh_crash_key = CRASH_ID if (CRASH_ID in vehicle.columns) else next((c for c in vehicles.columns if 'CRASH' in c), None)
veh_id_col = next((c for c in vehicle.columns if c in ['VEHICLE_ID','VEH_ID']), None)
pers_veh_col = next((c for c in people.columns if c in ['VEHICLE_ID','VEH_ID']), None)


if pers_veh_col and veh_id_col:
    join_on = [CRASH_ID, pers_veh_col] if (CRASH_ID in people.columns and CRASH_ID in vehicles.columns) else [pers_veh_col]
    pv = people.join(vehicles, on=join_on, how='left')
else:
    pv = people


if CRASH_ID and CRASH_ID in pv.columns and CRASH_ID in crashes.columns:
    pvc = pv.join(crashes, on=CRASH_ID, how='left')
else:
    pvc = pv


if 'CRASH_DATE_SOURCE' in pvc.columns and 'CRASH_DATETIME' not in pvc.columns:
    pvc = pvc.withColumn('CRASH_DATETIME', expr("try_to_timestamp(CRASH_DATE_SOURCE, 'MM/dd/yyyy hh:mm:ss a')"))


if 'CRASH_DATETIME' in pvc.columns:
    pvc = pvc.withColumn('HOUR', hour(col('CRASH_DATETIME')))
    pvc = pvc.withColumn('YEAR', year(col('CRASH_DATETIME')))


lon_col = next((c for c in ['LONGITUDE','X_COORDINATE','LON'] if c in pvc.columns), None)
if lon_col:
    pvc = pvc.withColumn('LON', col(lon_col).cast(DoubleType()))
lat_col = next((c for c in ['LATITUDE','Y_COORDINATE','LAT'] if c in pvc.columns), None)
if lat_col:
    pvc = pvc.withColumn('LAT', col(lat_col).cast(DoubleType()))


rows = pvc.count()
cols = len(pvc.columns)
print("Rows:", rows, "Cols:", cols)

sample_cols = [c for c in [state_col, age_col, 'ROADWAY_SURFACE_COND', 'PRIM_CONTRIBUTORY_CAUSE'] if c in pvc.columns]
if sample_cols:
    print(pvc.select(sample_cols).limit(5).toPandas())
else:
    print("No sample columns available to preview.")

from pyspark.sql.functions import col, when, upper, trim, lit, regexp_extract
from pyspark.sql.types import TimestampType
import json
import pandas as pd

def safe_to_pandas(spark_df, cols, limit=50000):
    projections = []
    for c in cols:
        # accept column-name strings or Column objects
        if isinstance(c, str):
            field = next((f for f in spark_df.schema.fields if f.name == c), None)
            if field is not None and isinstance(field.dataType, TimestampType):
                projections.append(col(c).cast('string').alias(c))
            elif c == 'CRASH_DATETIME':
                projections.append(col(c).cast('string').alias(c))
            else:
                projections.append(col(c))
        else:
            projections.append(c)
    try:
        return spark_df.select(*projections).limit(int(limit)).toPandas()
    except Exception:
        try:
            json_rows = spark_df.select(*projections).limit(int(limit)).toJSON().collect()
            return pd.DataFrame([json.loads(r) for r in json_rows])
        except Exception:
            return pd.DataFrame(columns=[c if isinstance(c, str) else str(c) for c in cols])

pvc_feat = pvc
fatal_col = next((c for c in pvc_feat.columns if c.upper() == 'INJURIES_FATAL'), None)
if fatal_col:
    pvc_feat = pvc_feat.withColumn('fatal_outcome', when(col(fatal_col).cast('double') > 0, True).otherwise(False))
else:
    if ('fatal' in globals()) and (CRASH_ID in fatal.columns) and (CRASH_ID in pvc_feat.columns):
        fatal_ids = set(fatal.select(CRASH_ID).dropna().toPandas()[CRASH_ID].astype(str))
        pvc_feat = pvc_feat.withColumn('fatal_outcome', col(CRASH_ID).cast('string').isin(list(fatal_ids)))
    else:
        pvc_feat = pvc_feat.withColumn('fatal_outcome', lit(False))

age_col_name = next((c for c in pvc_feat.columns if c in ['AGE','DRIVER_AGE','PERSON_AGE']), None)
if age_col_name:
    pvc_feat = pvc_feat.withColumn('DRIVER_AGE', col(age_col_name).cast('double'))
    pvc_feat = pvc_feat.withColumn(
        'age_bin',
        when(col('DRIVER_AGE') < 16, '<16')
        .when((col('DRIVER_AGE') >= 16) & (col('DRIVER_AGE') <= 24), '16-24')
        .when((col('DRIVER_AGE') >= 25) & (col('DRIVER_AGE') <= 34), '25-34')
        .when((col('DRIVER_AGE') >= 35) & (col('DRIVER_AGE') <= 49), '35-49')
        .when((col('DRIVER_AGE') >= 50) & (col('DRIVER_AGE') <= 64), '50-64')
        .when(col('DRIVER_AGE') >= 65, '65+')
        .otherwise('UNKNOWN')
    )
else:
    pvc_feat = pvc_feat.withColumn('age_bin', lit('UNKNOWN'))

state_col_name = next((c for c in pvc_feat.columns if c in ['DRIVER_LICENSE_STATE','DRVR_LIC_STATE','LIC_PLATE_STATE']), None)
if state_col_name:
    pvc_feat = pvc_feat.withColumn('license_state', upper(trim(col(state_col_name))))
else:
    pvc_feat = pvc_feat.withColumn('license_state', lit('UNKNOWN'))

surface_col = next((c for c in pvc_feat.columns if c in ['ROADWAY_SURFACE_COND','RDWY_SURF_COND']), None)
if surface_col:
    pvc_feat = pvc_feat.withColumn('surface', upper(trim(col(surface_col))))
else:
    pvc_feat = pvc_feat.withColumn('surface', lit('UNKNOWN'))

print("pvc_feat row count (before fixes):", pvc_feat.count())
try:
    sample_cols = list(pvc_feat.columns)[:10]
    print("pvc_feat sample (first cols):")
    print(safe_to_pandas(pvc_feat, sample_cols, limit=5))
except Exception as _:
    print("Could not collect pvc_feat sample")

if 'HOUR' not in pvc_feat.columns:
    if 'CRASH_DATETIME' in pvc_feat.columns:
        hour_ex = regexp_extract(col('CRASH_DATETIME'), r'(\d{1,2}):\d{2}:\d{2}\s*([AP]M)?', 1)
        ampm_ex = regexp_extract(col('CRASH_DATETIME'), r'(\d{1,2}):\d{2}:\d{2}\s*([AP]M)?', 2)
        pvc_feat = pvc_feat.withColumn(
            'HOUR',
            when(hour_ex == '', None).otherwise(
                when(ampm_ex == 'AM',
                     when(hour_ex == '12', lit(0)).otherwise(hour_ex.cast('int')))
                .when(ampm_ex == 'PM',
                     when(hour_ex == '12', lit(12)).otherwise(hour_ex.cast('int') + 12))
                .otherwise(hour_ex.cast('int'))
            )
        )
    elif 'CRASH_DATE' in pvc_feat.columns:
        hour_ex = regexp_extract(col('CRASH_DATE'), r'(\d{1,2}):\d{2}:\d{2}\s*([AP]M)?', 1)
        ampm_ex = regexp_extract(col('CRASH_DATE'), r'(\d{1,2}):\d{2}:\d{2}\s*([AP]M)?', 2)
        pvc_feat = pvc_feat.withColumn(
            'HOUR',
            when(hour_ex == '', None).otherwise(
                when(ampm_ex == 'AM',
                     when(hour_ex == '12', lit(0)).otherwise(hour_ex.cast('int')))
                .when(ampm_ex == 'PM',
                     when(hour_ex == '12', lit(12)).otherwise(hour_ex.cast('int') + 12))
                .otherwise(hour_ex.cast('int'))
            )
        )

if 'YEAR' not in pvc_feat.columns:
    if 'CRASH_DATE' in pvc_feat.columns:
        pvc_feat = pvc_feat.withColumn('YEAR', regexp_extract(col('CRASH_DATE'), r'(\d{4})', 1).cast('int'))
    elif 'CRASH_DATETIME' in pvc_feat.columns:
        pvc_feat = pvc_feat.withColumn('YEAR', regexp_extract(col('CRASH_DATETIME'), r'(\d{4})', 1).cast('int'))

lon_cand = next((c for c in pvc_feat.columns if c.upper() in ['LONGITUDE','X_COORDINATE','LON']), None)
if lon_cand and 'LON' not in pvc_feat.columns:
    pvc_feat = pvc_feat.withColumn('LON', col(lon_cand).cast('double'))
lat_cand = next((c for c in pvc_feat.columns if c.upper() in ['LATITUDE','Y_COORDINATE','LAT']), None)
if lat_cand and 'LAT' not in pvc_feat.columns:
    pvc_feat = pvc_feat.withColumn('LAT', col(lat_cand).cast('double'))

desired = [CRASH_ID, 'CRASH_DATETIME','HOUR','YEAR','LAT','LON','fatal_outcome','age_bin','license_state','surface','PRIM_CONTRIBUTORY_CAUSE']
cols_present = [c for c in desired if c in pvc_feat.columns]
print("Columns present for driver_df:", cols_present)
print("Rows after fixes:", pvc_feat.count())

if cols_present:
    driver_df = pvc_feat.select(*cols_present)
else:
    driver_df = pvc_feat.limit(0)

try:
    preview = safe_to_pandas(driver_df, cols_present, limit=5)
except Exception:
    preview = pd.DataFrame(columns=cols_present)
print(preview)


from pyspark.sql.functions import col, mean as spark_mean
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

fig, axes = plt.subplots(1, 3, figsize=(18,5))
age_order = ['<16','16-24','25-34','35-49','50-64','65+']
if 'age_bin' in driver_df.columns:
    age_agg = (driver_df.withColumn('fatal_outcome_num', col('fatal_outcome').cast('double'))
               .groupBy('age_bin')
               .agg(spark_mean('fatal_outcome_num').alias('fatal_rate'))
               .select(col('age_bin').cast('string').alias('age_bin'), col('fatal_rate')))
    try:
        age_rate_pd = age_agg.toPandas()
    except Exception:
        try:
            age_rate_pd = safe_to_pandas(age_agg, ['age_bin','fatal_rate'], limit=10000)
        except Exception:
            rows = age_agg.collect()
            age_rate_pd = pd.DataFrame([r.asDict() for r in rows])

    if 'age_bin' not in age_rate_pd.columns and len(age_rate_pd.columns) >= 2:
        age_rate_pd.columns = ['age_bin','fatal_rate'] + list(age_rate_pd.columns[2:])

    age_map = {str(k).strip(): float(v) if v is not None else 0.0 for k, v in zip(age_rate_pd['age_bin'].astype(str), age_rate_pd['fatal_rate'])} if not age_rate_pd.empty else {}
    y = [age_map.get(bin_label, 0.0) for bin_label in age_order]

    if any(v > 0 for v in y):
        sns.barplot(x=age_order, y=y, ax=axes[0], color='tab:blue')
        axes[0].set_title('Fatality rate by driver age bin')
        axes[0].set_xlabel('Age bin')
        axes[0].set_ylabel('Fatality rate')
    else:
        axes[0].set_visible(False)
else:
    axes[0].set_visible(False)

if 'license_state' in driver_df.columns:
    state_counts = driver_df.groupBy('license_state').count().orderBy(col('count').desc()).limit(10).select(col('license_state').cast('string').alias('license_state'))
    try:
        top_states = state_counts.toPandas()['license_state'].astype(str).tolist()
    except Exception:
        try:
            top_states = safe_to_pandas(state_counts, ['license_state'], limit=1000)['license_state'].astype(str).tolist()
        except Exception:
            top_states = [r['license_state'] for r in state_counts.collect()]
    state_agg = (driver_df.filter(col('license_state').isin(top_states))
                 .withColumn('fatal_outcome_num', col('fatal_outcome').cast('double'))
                 .groupBy('license_state')
                 .agg(spark_mean('fatal_outcome_num').alias('fatal_rate'))
                 .select(col('license_state').cast('string').alias('license_state'), col('fatal_rate')))
    try:
        state_rate_df = state_agg.toPandas().sort_values('fatal_rate', ascending=False)
    except Exception:
        try:
            state_rate_df = safe_to_pandas(state_agg, ['license_state','fatal_rate'], limit=1000).sort_values('fatal_rate', ascending=False)
        except Exception:
            state_rate_df = pd.DataFrame([r.asDict() for r in state_agg.collect()]).sort_values('fatal_rate', ascending=False)
    sns.barplot(x=state_rate_df['fatal_rate'], y=state_rate_df['license_state'], ax=axes[1], color='tab:purple', orient='h')
    axes[1].set_title('Fatality rate by license state (top 10)')
    axes[1].set_xlabel('Fatality rate')
else:
    axes[1].set_visible(False)

if 'surface' in driver_df.columns:
    surf_agg = (driver_df.withColumn('fatal_outcome_num', col('fatal_outcome').cast('double'))
                .groupBy('surface')
                .agg(spark_mean('fatal_outcome_num').alias('fatal_rate'))
                .select(col('surface').cast('string').alias('surface'), col('fatal_rate')))
    try:
        surf_rate_df = surf_agg.toPandas().sort_values('fatal_rate', ascending=False).head(10)
    except Exception:
        try:
            surf_rate_df = safe_to_pandas(surf_agg, ['surface','fatal_rate'], limit=1000).sort_values('fatal_rate', ascending=False).head(10)
        except Exception:
            surf_rate_df = pd.DataFrame([r.asDict() for r in surf_agg.collect()]).sort_values('fatal_rate', ascending=False).head(10)
    sns.barplot(x=surf_rate_df['fatal_rate'], y=surf_rate_df['surface'], ax=axes[2], color='tab:orange', orient='h')
    axes[2].set_title('Fatality rate by roadway surface (top 10)')
    axes[2].set_xlabel('Fatality rate')
else:
    axes[2].set_visible(False)

plt.tight_layout()
plt.savefig('fatality_rate_trends_patterns.png')

from pyspark.sql.functions import col, mean as spark_mean
import folium
from folium.plugins import HeatMap
from folium import FeatureGroup, CircleMarker
import pandas as pd

AGE_MARKER_CAP = 3000
SURFACE_HEAT_CAP = 40000

if 'LAT' in driver_df.columns and 'LON' in driver_df.columns and 'fatal_outcome' in driver_df.columns:
    fat_df = driver_df.filter(col('LAT').isNotNull() & col('LON').isNotNull() & (col('fatal_outcome') == True))
else:
    fat_df = None

if fat_df is None or fat_df.count() == 0:
    print('No fatal crashes with coordinates available for map.')
else:
    try:
        lat_med = fat_df.approxQuantile('LAT', [0.5], 0.01)[0]
        lon_med = fat_df.approxQuantile('LON', [0.5], 0.01)[0]
        center = [float(lat_med), float(lon_med)]
    except Exception:
        sample_pd = fat_df.select('LAT','LON').limit(1000).toPandas()
        if sample_pd.empty:
            print('No valid coordinates after sampling; skipping map.')
            center = [0.0,0.0]
        else:
            center = [float(sample_pd['LAT'].median()), float(sample_pd['LON'].median())]

    fmap = folium.Map(location=center, zoom_start=11, tiles=TILE_URL, attr='OSM')

    age_bins = ['<16','16-24','25-34','35-49','50-64','65+']
    colors = {
        '<16':'#984ea3','16-24':'#e41a1c','25-34':'#377eb8',
        '35-49':'#4daf4a','50-64':'#ff7f00','65+':'#a65628'
    }
    for ab in age_bins:
        layer = FeatureGroup(name=f"Fatal drivers: Age {ab}")
        try:
            sub_spark = fat_df.filter(col('age_bin').cast('string') == ab)
            try:
                sub_pd = sub_spark.select('LAT','LON','CRASH_DATETIME').limit(AGE_MARKER_CAP).toPandas()
            except Exception:
                try:
                    sub_pd = safe_to_pandas(sub_spark, ['LAT','LON','CRASH_DATETIME'], limit=AGE_MARKER_CAP)
                except Exception:
                    sub_pd = pd.DataFrame(columns=['LAT','LON','CRASH_DATETIME'])
            if not sub_pd.empty:
                for _, r in sub_pd.iterrows():
                    try:
                        folium.CircleMarker([float(r['LAT']), float(r['LON'])], radius=4, color=colors.get(ab,'#444'),
                                            fill=True, fill_opacity=0.8,
                                            popup=f"Age {ab} | {r.get('CRASH_DATETIME','')}").add_to(layer)
                    except Exception:
                        continue
        except Exception:
            pass
        layer.add_to(fmap)

 
    if 'surface' in driver_df.columns:
        try:
            surf_rates = (driver_df.withColumn('fatal_num', col('fatal_outcome').cast('double'))
                          .groupBy('surface')
                          .agg(spark_mean('fatal_num').alias('rate'))
                          .orderBy(col('rate').desc())
                          .limit(5))
            try:
                top_surfaces = [s for s in surf_rates.select('surface').toPandas()['surface'].astype(str).tolist()]
            except Exception:
                top_surfaces = [r['surface'] for r in surf_rates.collect()]
        except Exception:
            top_surfaces = []
    else:
        top_surfaces = []

    for s in top_surfaces:
        layer = FeatureGroup(name=f"Fatal: Surface {s}")
        try:
            sub_spark = fat_df.filter(col('surface').cast('string') == s)
            try:
                heat_pd = sub_spark.select('LAT','LON').limit(SURFACE_HEAT_CAP).toPandas().dropna()
            except Exception:
                try:
                    heat_pd = safe_to_pandas(sub_spark, ['LAT','LON'], limit=SURFACE_HEAT_CAP).dropna()
                except Exception:
                    heat_pd = pd.DataFrame(columns=['LAT','LON'])
            if not heat_pd.empty:
                HeatMap(heat_pd[['LAT','LON']].values.tolist(), radius=12, blur=14, max_zoom=13).add_to(layer)
        except Exception:
            pass
        layer.add_to(fmap)

    folium.LayerControl(collapsed=False).add_to(fmap)
    out_path = 'fatal_policy_map_pyspark.html'
    fmap.save(str(out_path))
    print(f'Fatal policy map saved to {out_path}')





from pyspark.sql.functions import col, when, upper, trim, lit, hour, date_format, coalesce, expr
from pyspark.sql.types import TimestampType
import json

cr2 = combined_clean

# If CRASH_DATETIME missing, parse tolerant CRASH_DATE -> CRASH_DATETIME
if 'CRASH_DATE' in cr2.columns and 'CRASH_DATETIME' not in cr2.columns:
    cr2 = cr2.withColumn('CRASH_DATETIME', expr("try_to_timestamp(CRASH_DATE, 'MM/dd/yyyy hh:mm:ss a')"))

fatal_flag_col = next((c for c in cr2.columns if c.upper() == 'INJURIES_FATAL'), None)
if fatal_flag_col:
    cr2 = cr2.withColumn('is_fatal', when(col(fatal_flag_col).cast('double') > 0, True).otherwise(False))
else:
    if ('fatal' in globals()) and (CRASH_ID in fatal.columns) and (CRASH_ID in cr2.columns):
        fatal_ids_df = (fatal
                        .select(col(CRASH_ID).alias(CRASH_ID))
                        .dropna()
                        .distinct()
                        .withColumn('is_fatal_ref', lit(True)))
        cr2 = cr2.join(fatal_ids_df, on=CRASH_ID, how='left')
        cr2 = cr2.withColumn('is_fatal', coalesce(col('is_fatal_ref'), lit(False))).drop('is_fatal_ref')
    else:
        cr2 = cr2.withColumn('is_fatal', lit(False))

if 'CRASH_DATETIME' in cr2.columns:
    cr2 = cr2.withColumn('HOUR', hour(col('CRASH_DATETIME')))
    cr2 = cr2.withColumn('DAY_OF_WEEK', date_format(col('CRASH_DATETIME'), 'EEEE'))

for cat_col in ['LIGHTING_CONDITION','WEATHER_CONDITION','ROADWAY_SURFACE_COND','PRIM_CONTRIBUTORY_CAUSE']:
    if cat_col in cr2.columns:
        cr2 = cr2.withColumn(cat_col, upper(trim(col(cat_col))))

print("Rows:", cr2.count(), "Cols:", len(cr2.columns))
preview_cols = [c for c in ['is_fatal','HOUR','DAY_OF_WEEK','CRASH_DATETIME'] if c in cr2.columns]

if preview_cols:
    proj = []
    for c in preview_cols:
        field = next((f for f in cr2.schema.fields if f.name == c), None)
        if field is not None and isinstance(field.dataType, TimestampType):
            proj.append(col(c).cast('string').alias(c))
        else:
            proj.append(col(c))

    try:
        print(cr2.select(*proj).limit(5).toPandas())
    except Exception:
        try:
            json_rows = cr2.select(*proj).limit(5).toJSON().collect()
            parsed = []
            for j in json_rows:
                try:
                    parsed.append(json.loads(j))
                except Exception:
                    parsed.append(j)
            print(parsed)
        except Exception as e:
            print("Preview failed:", str(e))

from pyspark.sql.functions import col, sum as spark_sum, count as spark_count, mean as spark_mean, desc, row_number
from pyspark.sql import Window

fig, axes = plt.subplots(2, 3, figsize=(18,10))

if 'HOUR' in cr2.columns:
    hr_df = cr2.filter(col('HOUR').isNotNull()).withColumn('HOUR', col('HOUR').cast('int')).withColumn('fatal_num', col('is_fatal').cast('int'))
    hourly = hr_df.groupBy('HOUR').agg(
        spark_count('*').alias('total'),
        spark_sum('fatal_num').alias('fatal')
    ).orderBy('HOUR')
    try:
        hourly_pd = hourly.toPandas()
    except Exception:
        hourly_pd = safe_to_pandas(hourly, ['HOUR','total','fatal'], limit=10000)
    if isinstance(hourly_pd, (pd.DataFrame,)) and not hourly_pd.empty:
        hourly_pd = hourly_pd.sort_values('HOUR')
        hourly_pd['nonfatal'] = hourly_pd['total'] - hourly_pd['fatal']
        sns.lineplot(x=hourly_pd['HOUR'], y=(hourly_pd['fatal'] / hourly_pd['total']).fillna(0), ax=axes[0,0], label='fatal_rate', color='C3')
        sns.lineplot(x=hourly_pd['HOUR'], y=(hourly_pd['nonfatal'] / hourly_pd['total']).fillna(0), ax=axes[0,0], label='nonfatal_rate', color='C0')
        axes[0,0].set_title('Hour of crash (fatal vs nonfatal rate)')
        axes[0,0].set_xlabel('Hour')
        axes[0,0].legend()
    else:
        axes[0,0].set_visible(False)
else:
    axes[0,0].set_visible(False)

def plot_top_cat(col_name, ax, color='tab:olive', topk=8, orient='h', title=None):
    if col_name not in cr2.columns:
        ax.set_visible(False)
        return
    top_df = cr2.groupBy(col_name).count().orderBy(desc('count')).limit(topk)
    try:
        top_list = safe_to_pandas(top_df.select(col_name), [col_name], limit=1000)[col_name].astype(str).tolist()
    except Exception:
        top_list = [r[col_name] for r in top_df.collect()]
    if not top_list:
        ax.set_visible(False); return
    agg = (cr2.filter(col(col_name).isin(top_list))
           .withColumn('fatal_num', col('is_fatal').cast('double'))
           .groupBy(col_name)
           .agg(spark_mean('fatal_num').alias('fatal_rate')))
    try:
        agg_pd = agg.toPandas().sort_values('fatal_rate', ascending=False)
    except Exception:
        agg_pd = safe_to_pandas(agg, [col_name,'fatal_rate'], limit=2000).sort_values('fatal_rate', ascending=False)
    if agg_pd.empty:
        ax.set_visible(False)
        return
    sns.barplot(x=agg_pd['fatal_rate'], y=agg_pd[col_name], ax=ax, color=color, orient=orient)
    if title:
        ax.set_title(title)

# for Light
plot_top_cat('LIGHTING_CONDITION', axes[0,1], color='tab:olive', title='Fatality rate by lighting (top)')

# for Weather
plot_top_cat('WEATHER_CONDITION', axes[0,2], color='tab:cyan', title='Fatality rate by weather (top)')

# for Surface
surface_col = 'ROADWAY_SURFACE_COND' if 'ROADWAY_SURFACE_COND' in cr2.columns else None
if surface_col:
    plot_top_cat(surface_col, axes[1,0], color='tab:orange', title='Fatality rate by surface (top)')
else:
    axes[1,0].set_visible(False)

veh_type_col = next((c for c in vehicles.columns if c.upper() in ['VEHICLE_TYPE','VEHICLE_MAKE','VEHICLE_MODEL','BODY_TYPE']), None)
if veh_type_col and CRASH_ID in vehicles.columns and CRASH_ID in cr2.columns:
    vv = vehicles.select([col(c).alias(c.upper()) for c in vehicles.columns]) if hasattr(vehicles, 'columns') else vehicles
    vt_col = veh_type_col if veh_type_col in vv.columns else veh_type_col.upper()
    counts = vv.groupBy(CRASH_ID, vt_col).count()
    w = Window.partitionBy(CRASH_ID).orderBy(desc('count'))
    top_per_crash = (counts
                     .withColumn('rn', row_number().over(w))
                     .filter(col('rn') == 1)
                     .select(CRASH_ID, col(vt_col).alias('VEHICLE_TYPE_MODE')))
   
    cr_with_mode = cr2.join(top_per_crash, on=CRASH_ID, how='left')
    mode_counts = cr_with_mode.groupBy('VEHICLE_TYPE_MODE').count().orderBy(desc('count')).limit(8)
    try:
        top_modes = safe_to_pandas(mode_counts.select('VEHICLE_TYPE_MODE'), ['VEHICLE_TYPE_MODE'], limit=1000)['VEHICLE_TYPE_MODE'].astype(str).tolist()
    except Exception:
        top_modes = [r['VEHICLE_TYPE_MODE'] for r in mode_counts.collect()]
    if top_modes:
        agg = (cr_with_mode.filter(col('VEHICLE_TYPE_MODE').isin(top_modes))
               .withColumn('fatal_num', col('is_fatal').cast('double'))
               .groupBy('VEHICLE_TYPE_MODE')
               .agg(spark_mean('fatal_num').alias('fatal_rate')))
        try:
            agg_pd = agg.toPandas().sort_values('fatal_rate', ascending=False)
        except Exception:
            agg_pd = safe_to_pandas(agg, ['VEHICLE_TYPE_MODE','fatal_rate'], limit=2000).sort_values('fatal_rate', ascending=False)
        if not agg_pd.empty:
            sns.barplot(x=agg_pd['fatal_rate'], y=agg_pd['VEHICLE_TYPE_MODE'], ax=axes[1,1], color='tab:red', orient='h')
            axes[1,1].set_title('Fatality rate by vehicle type (mode)')
        else:
            axes[1,1].set_visible(False)
    else:
        axes[1,1].set_visible(False)
else:
    axes[1,1].set_visible(False)

age_order = ['<16','16-24','25-34','35-49','50-64','65+']
if 'age_bin' in driver_df.columns:
    age_agg = (driver_df
               .withColumn('age_bin_str', col('age_bin').cast('string'))
               .withColumn('fatal_num', col('fatal_outcome').cast('double'))
               .groupBy('age_bin_str')
               .agg(spark_mean('fatal_num').alias('fatal_rate')))
    try:
        age_pd = age_agg.toPandas()
    except Exception:
        age_pd = safe_to_pandas(age_agg, ['age_bin_str','fatal_rate'], limit=10000)
    if isinstance(age_pd, (pd.DataFrame,)) and not age_pd.empty:
        age_pd['age_bin_str'] = age_pd['age_bin_str'].astype(str).str.strip()
        age_map = dict(zip(age_pd['age_bin_str'], age_pd['fatal_rate'].astype(float)))
        x = age_order
        y = [age_map.get(k, 0.0) for k in x]
        if any(v > 0 for v in y):
            sns.barplot(x=x, y=y, ax=axes[1,2], color='tab:blue')
            axes[1,2].set_title('Driver age fatality rate')
            axes[1,2].set_xlabel('Age bin')
            axes[1,2].set_ylabel('Fatality rate')
        else:
            axes[1,2].set_visible(False)
    else:
        axes[1,2].set_visible(False)
else:
    axes[1,2].set_visible(False)

plt.tight_layout()
plt.savefig('crash_distributions.png')
print("I reached end without any errors!")
