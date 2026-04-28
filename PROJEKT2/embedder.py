# Notebook: gold
# Path: /Users/mozarjunior13@gmail.com/PROJECT2/gold



# ── Config ────────────────────────────────────────────────────────────────────
CATALOG       = "proj2_catalog"
SCHEMA        = "default"
SILVER_TABLE  = f"{CATALOG}.{SCHEMA}.silver_chunks" # poprzednia tabelka jest uzyta 
GOLD_TABLE    = f"{CATALOG}.{SCHEMA}.gold_embeddings"

MODEL_NAME    = "all-MiniLM-L6-v2"   # klasyczny model transformer.
BATCH_SIZE    = 64

spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"USE SCHEMA {SCHEMA}")


# PART 1 — tworze embeddingi, dodaje do tabelki silver (ale tylko wewnątrz tego pliku)


from sentence_transformers import SentenceTransformer
import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, FloatType

print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)


silver_pd = spark.table(SILVER_TABLE).toPandas()
print(f"Embedding {len(silver_pd)} chunks...")

# Embedding z normalizacją
embeddings = model.encode(
    silver_pd["chunk_text"].tolist(),
    batch_size   = BATCH_SIZE,
    show_progress_bar = True,
    normalize_embeddings = True,   
)


silver_pd["embedding"] = embeddings.tolist()


gold_df = spark.createDataFrame(silver_pd) # nowy spark df


gold_df = gold_df.withColumn(
    "embedding",
    F.col("embedding").cast(ArrayType(FloatType()))
)

(gold_df
 .write
 .format("delta")
 .mode("overwrite")
 .option("overwriteSchema", "true")
 .saveAsTable(GOLD_TABLE))

print(f"\nGOLD written → {GOLD_TABLE}")
print(f"Embedding dimension: {len(embeddings[0])}")
spark.table(GOLD_TABLE).select(
    "chunk_id", "filename", "page_number", "word_count"
).show(5)
