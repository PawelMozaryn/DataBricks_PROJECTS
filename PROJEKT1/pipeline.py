# Notebook: pipeline
# Path: /Users/mozarjunior13@gmail.com/PROJECT1/pipeline
#
# UWAGA! To jest Delta Table declarative pipeline. Nie do puszczania jako osoby plik!


import dlt
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType
import re

# ── Config ───────────────────────────────────────────────────────────────────
CATALOG     = "proj1_catalog"
VOLUME_PATH = f"/Volumes/{CATALOG}/default/DATASETS"

## JAKIE PLIKI MAJA FORMAT datasetN.csv??

import subprocess, os

def _list_datasets():
    """Return sorted list of (t:int, path:str) for every datasetN.csv found."""
    try:
        files = dbutils.fs.ls(VOLUME_PATH)   # noqa: F821  (dbutils injected by DLT)
        result = []
        for f in files:
            m = re.match(r"dataset(\d+)\.csv$", f.name)
            if m:
                result.append((int(m.group(1)), f.path))
        return sorted(result)
    except Exception:
        return []

DATASETS = _list_datasets()


# ═════════════════════════════════════════════════════════════════════════════
# BRONZE  —  ladowanie surowych nie przetransformowanych danych !
# ═════════════════════════════════════════════════════════════════════════════

def _make_bronze(t: int, path: str):
    """Factory: returns a DLT streaming-table function for dataset t."""

    @dlt.table( # DZIEKI TEMU GRAF PIPELINU WIE NA JAKIM JESTEM ETAPIE 
        name=f"bronze_{t}",
        comment=f"Raw ingestion of dataset{t}.csv",
        table_properties={"quality": "bronze"}, # w datalake ta tabelka jest brązowa
    )
    def _bronze():
        return (
            spark.read                          # noqa: F821
            .option("header", "true")
            .option("inferSchema", "true")
            .csv(path)
            .withColumn("_ingestion_ts", F.current_timestamp())
            .withColumn("_t", F.lit(t))
        )

    return _bronze


### Dla kazdego datasetu chcę dokladnie jedną tabelkę.
for t, path in DATASETS:
    _make_bronze(t, path)


# ═════════════════════════════════════════════════════════════════════════════
# SILVER  —  clean + encode, one table per dataset
# ═════════════════════════════════════════════════════════════════════════════

def _make_silver(t: int):

    @dlt.table(
        name=f"silver_{t}",
        comment=f"Cleaned dataset{t}",
        table_properties={"quality": "silver"},
    )
    def _silver():
        df = dlt.read(f"bronze_{t}")

        # 1. USUWAM BRAKI DANYCH 


        df = df.dropna()

        # 2. CHCE LUDZI O WYSOKOSCI Z [10, 90]


        df = df.filter((F.col("Height") >= 10) & (F.col("Height") <= 90))

        # 3. ONEHOT ENCODING


        for level in ["MSc", "Ba", "PhD"]:
            df = df.withColumn(
                f"Edu_{level}",
                (F.col("Edu") == level).cast(IntegerType())
            )
        df = df.drop("Edu")

        # 4. ROZBIJAM KOLUMNĘ Z DATA NA MIESIAC I ROK
        df = (df
              .withColumn("Month", F.month(F.to_date("date", "yyyy-MM-dd")))
              .withColumn("Year",  F.year (F.to_date("date", "yyyy-MM-dd")))
              .drop("date"))

        # 5. ZMIENNA SEX BINARNA
        df = df.withColumn("Sex_M", (F.col("Sex") == "M").cast(IntegerType())).drop("Sex")

        return df

    return _silver


for t, _ in DATASETS:
    _make_silver(t)


# ═════════════════════════════════════════════════════════════════════════════
# GOLD  —  TABELKA POD ML - zmienne gotowe do modelu regresji liniowej.
# ═════════════════════════════════════════════════════════════════════════════

def _make_gold(t: int):

    @dlt.table(
        name=f"gold_{t}",
        comment=f"Regression-ready dataset{t} (numeric only)",
        table_properties={"quality": "gold"},
    )
    def _gold():
        df = dlt.read(f"silver_{t}")
        # USUWAM KOLUMNY Z METADATA, NIENUMERYCZNE, NIEZAWIERAJACE INFORMACJI O ZACHOWANIU ZMIENNEJ Y.

        drop_cols = ["Person_ID", "_ingestion_ts", "_t",
                     "Month", "Year"]   
        for c in drop_cols:
            if c in df.columns:
                df = df.drop(c)
    
        for c in df.columns:
            df = df.withColumn(c, F.col(c).cast(DoubleType()))
        return df

    return _gold


for t, _ in DATASETS:
    _make_gold(t)

