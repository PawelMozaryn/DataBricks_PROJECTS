# Notebook: proj2_pipeline
# Path: /Users/mozarjunior13@gmail.com/PROJECT2/proj2_pipeline


import dlt
import fitz                       
from datetime import datetime
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType, StructType, StructField,
    StringType, IntegerType
)

# ── Config ────────────────────────────────────────────────────────────────────
CATALOG    = "proj2_catalog"
SCHEMA     = "default"
SOURCE_VOL = f"/Volumes/{CATALOG}/{SCHEMA}/text_pdfs"

CHUNK_SIZE = 150 # Chunki mają miec 150 slow
OVERLAP    = 30 # nachodzą na siebie po 30 slow, aby nie gubic kontektu

# ══════════════════════════════════════════════════════════════════════════════
# BRONZE — tabela zawiera 1 wiersz na pdf (nazwa, dlugi tekst)
# ══════════════════════════════════════════════════════════════════════════════

@dlt.table(
    name    = "bronze_pages",
    comment = "Raw text extracted from PDFs, one row per page",
    table_properties = {"quality": "bronze"},
)
def bronze_pages():
    # Discover PDFs from the volume
    pdf_files = [
        f.path for f in dbutils.fs.ls(SOURCE_VOL)   # noqa: F821
        if f.name.endswith(".pdf")
    ]

    rows = []
    ingestion_ts = datetime.utcnow().isoformat()

    for pdf_path in pdf_files:
        filename   = pdf_path.split("/")[-1]
        local_path = pdf_path.replace("dbfs:", "")

        doc = fitz.open(local_path)
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if not text:
                continue
            rows.append((
                filename,
                page_num + 1,
                len(doc),
                text,
                len(text),
                ingestion_ts,
            ))
        doc.close()

    schema = """
        filename     STRING,
        page_number  INT,
        total_pages  INT,
        raw_text     STRING,
        char_count   INT,
        ingestion_ts STRING
    """
    return spark.createDataFrame(rows, schema=schema)   # noqa: F821


# ══════════════════════════════════════════════════════════════════════════════
# SILVER — tabela zawiera chunki + info z jakiego tekstu są, tworzy srebrną tabelę
# ══════════════════════════════════════════════════════════════════════════════

def _chunk_text(text: str, chunk_size: int, overlap: int):
    """Split text into overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start  = 0
    idx    = 0
    step   = chunk_size - overlap
    while start < len(words):
        end         = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append((idx, " ".join(chunk_words), len(chunk_words), start, end - 1))
        if end == len(words):
            break
        start += step
        idx   += 1
    return chunks


_chunk_schema = ArrayType(StructType([
    StructField("chunk_index", IntegerType()),
    StructField("chunk_text",  StringType()),
    StructField("word_count",  IntegerType()),
    StructField("start_word",  IntegerType()),
    StructField("end_word",    IntegerType()),
]))

_chunk_udf = F.udf(
    lambda text: _chunk_text(text, CHUNK_SIZE, OVERLAP),
    _chunk_schema,
)


@dlt.table(
    name    = "silver_chunks",
    comment = "Text chunked into 150-word windows with 30-word overlap",
    table_properties = {"quality": "silver"},
)
def silver_chunks():
    return (
        dlt.read("bronze_pages")
        .withColumn("chunk", F.explode(_chunk_udf(F.col("raw_text"))))
        .select(
            F.concat_ws("_",
                F.col("filename"),
                F.col("page_number").cast("string"),
                F.col("chunk.chunk_index").cast("string"),
            ).alias("chunk_id"),
            F.col("filename"),
            F.col("page_number"),
            F.col("total_pages"),
            F.col("chunk.chunk_index").alias("chunk_index"),
            F.col("chunk.chunk_text").alias("chunk_text"),
            F.col("chunk.word_count").alias("word_count"),
            F.col("chunk.start_word").alias("start_word"),
            F.col("chunk.end_word").alias("end_word"),
            F.col("ingestion_ts"),
        )
    )
