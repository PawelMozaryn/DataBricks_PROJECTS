# Notebook: stats_job
# Path: /Users/mozarjunior13@gmail.com/PROJECT1/stats_job


import math
import re
from pyspark.ml.feature    import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql           import functions as F
from pyspark.sql.types     import DoubleType

# ── KONFIGURACJA ───────────────────────────────────────────────────────────────────
CATALOG        = "proj1_catalog"
SCHEMA         = "default"       
SUMMARY_SCHEMA = "summary"      
TARGET         = "Y"

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SUMMARY_SCHEMA}")
spark.sql(f"USE SCHEMA {SCHEMA}")

# ── To etap po data-processingu, zgodnie z modelem medallion korzystamy ze złotych tabel ───────────────────────────────────────────────
all_tables = [r.tableName for r in spark.sql("SHOW TABLES").collect()]
gold_tables = sorted(
    [t for t in all_tables if re.match(r"^gold_\d+$", t)],
    key=lambda t: int(t.split("_")[1])
)

if not gold_tables:
    raise RuntimeError("No gold_t tables found. Run the DLT pipeline first.")

print(f"Found gold tables: {gold_tables}")

# ──  Trenuję regresję na kazdej ze zlotych tabel, zbieram wyniki w tabeli  ────────────────────────────
rows = []

for table_name in gold_tables:
    t = int(table_name.split("_")[1])

    gold_df  = spark.table(f"{CATALOG}.{SCHEMA}.{table_name}")
    silver_df = spark.table(f"{CATALOG}.{SCHEMA}.silver_{t}")
    bronze_df = spark.table(f"{CATALOG}.{SCHEMA}.bronze_{t}")

    # ile rekordów 
    n_gold   = gold_df.count()
    n_silver = silver_df.count()
    n_bronze = bronze_df.count()
    pct_silver = round(100.0 * n_silver / n_bronze, 2) if n_bronze > 0 else None

    # pyspark ml potrzebuje Doubli, dlatego upewniam sie ze tak jest 
    for c in gold_df.columns:
        gold_df = gold_df.withColumn(c, F.col(c).cast(DoubleType()))
    gold_df = gold_df.dropna() # upewniam sie ze brak NA

    feature_cols = [c for c in gold_df.columns if c != TARGET]

    assembled = (
        VectorAssembler(inputCols=feature_cols, outputCol="features")
        .transform(gold_df)
        .select("features", TARGET)
    )

    model   = LinearRegression(featuresCol="features", labelCol=TARGET,
                               maxIter=200, regParam=0.0).fit(assembled)
    summary = model.summary

    mse           = round(summary.meanSquaredError, 4) # MSE
    var_estimate  = round(mse * n_gold / (n_gold - len(feature_cols)), 4) # estymator nieobciazony wariancji SSE / (n-p)   
    true_variance = round(t, 4) # prawdziwa wariancja

    rows.append((t, n_bronze, n_silver, n_gold,
                 pct_silver, mse, var_estimate, true_variance)) # tworzę tabelkę wynikową 

    print(f"  t={t}  n_gold={n_gold}  MSE={mse:.4f}  "
          f"true_var={true_variance:.4f}")

## Tworzę tabelkę proj1_catalog.summary.stats
stats_schema = """
    t             INT,
    n_bronze      LONG,
    n_silver      LONG,
    n_gold        LONG,
    pct_silver    DOUBLE,
    mse           DOUBLE,
    var_estimate  DOUBLE,
    true_variance DOUBLE
"""

(spark
 .createDataFrame(rows, schema=stats_schema)
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOG}.{SUMMARY_SCHEMA}.stats"))

print(f"stats written in {CATALOG}.{SUMMARY_SCHEMA}.stats")
spark.table(f"{CATALOG}.{SUMMARY_SCHEMA}.stats").orderBy("t").show()

# Tworzę tabelkę gold_main, która zawiera całość złotych danych, ale po oczyszczeniu!
from functools import reduce
from pyspark.sql import DataFrame

gold_dfs = []
for table_name in gold_tables:
    t = int(table_name.split("_")[1])
    df = (spark
          .table(f"{CATALOG}.{SCHEMA}.{table_name}")
          .withColumn("t", F.lit(t)))
    gold_dfs.append(df)

gold_main = reduce(DataFrame.unionByName, gold_dfs)

(gold_main
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .partitionBy("t")
 .saveAsTable(f"{CATALOG}.{SUMMARY_SCHEMA}.gold_main"))

print(f"gold_main written → {CATALOG}.{SUMMARY_SCHEMA}.gold_main  "
      f"({gold_main.count()} rows, partitioned by t)")
