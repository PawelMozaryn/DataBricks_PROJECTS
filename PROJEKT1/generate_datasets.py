# Notebook: generate_datasets
# Path: /Users/mozarjunior13@gmail.com/PROJECT1/generate_datasets


# ── 0. Config ────────────────────────────────────────────────────────────────
T = 20       # tyle chce datasetow, t = 1, ..., T
CATALOG = "proj1_catalog"
VOLUME  = "DATASETS"          # Volume name  (proj1_catalog/default/DATASETS)
VOLUME_PATH = f"/Volumes/{CATALOG}/default/{VOLUME}"

import numpy as np
import pandas as pd
import uuid, os
from datetime import date, timedelta
import random

# ── 1. Ustalone beta (bez meta zmiennych takich jak id i data) ──────────────────────────────
# Cechy: Age, Height, Weight, TotalMoney, Edu_Ba, Edu_MSc, Edu_PhD,
#                    Sex_M, Avgbpm
BETA = np.array([
     0.15,   #  Age
    -0.30,   #  Height
     0.20,   #  Weight
     0.00050,#  TotalMoney
     1.50,   #  Edu_Ba   
     2.80,   #  Edu_MSc
     0.00,   #  Edu_PhD
     3.00,   #  Sex_M  (1 = Male, 0 = Female)
    -0.10,   #  Avgbpm
])
INTERCEPT = 10.0

EDU_LEVELS = ["MSc", "Ba", "PhD"]
START_DATE = date(2023, 1, 1)

# ── 2. Funkcja budująca ramke danych ───────────────────────────────────────

def make_dataset(t: int, rng: np.random.Generator) -> pd.DataFrame:
    n = rng.integers(500, 1001)          # ~ U({500,...,1000})
    sigma = float(t ** 0.5)            # to jest odchylenie standardowe bledu, variancja to t

    # Raw features
    age        = rng.integers(18, 70,  size=n).astype(float)
    height     = rng.integers(50, 210, size=n).astype(float)   # cm
    weight     = rng.integers(40, 130, size=n).astype(float)   # kg
    total_money= rng.uniform(0, 100_000, size=n)
    edu        = rng.choice(EDU_LEVELS, size=n)
    sex        = rng.choice(["M", "F"], size=n)
    avgbpm     = rng.integers(50, 110, size=n).astype(float)
    person_id  = [str(uuid.uuid4()) for _ in range(n)]
    rec_date   = [(START_DATE + timedelta(days=int((t-1)*30 + rng.integers(0, 30)))).isoformat()
                  for _ in range(n)]

    # Onehot encoding zmiennych kategorialnych
    edu_ba  = (edu == "Ba" ).astype(float)
    edu_msc = (edu == "MSc").astype(float)
    edu_phd = (edu == "PhD").astype(float)
    sex_m   = (sex == "M"  ).astype(float)

    X = np.column_stack([
        age, height, weight, total_money,
        edu_ba, edu_msc, edu_phd,
        sex_m, avgbpm
    ])
    eps = rng.normal(0, sigma, size=n)
    Y   = INTERCEPT + X @ BETA + eps

    df = pd.DataFrame({
        "Age"        : age,
        "Height"     : height,
        "Weight"     : weight,
        "TotalMoney" : total_money,
        "Edu"        : edu,
        "Sex"        : sex,
        "Avgbpm"     : avgbpm,
        "Person_ID"  : person_id,
        "date"       : rec_date,
        "Y"          : Y,
    })

    # ── Dodaję okolo 3% braków danych ────────────────────────────
    na_cols = ["Age", "Height", "Weight", "TotalMoney", "Avgbpm", "Y"]
    na_mask = rng.random(size=(n, len(na_cols))) < 0.03
    for j, col in enumerate(na_cols):
        df.loc[na_mask[:, j], col] = np.nan

    # ── Dodaję 1% outlierów wysokości ──────────────
    outlier_idx = rng.choice(n, size=max(1, n // 100), replace=False)
    df.loc[outlier_idx, "Height"] = rng.choice(
        list(range(0, 10)) + list(range(91, 220)),
        size=len(outlier_idx)
    ).astype(float)

    return df


# ── 3. Zapisuje pliki csv w zadanym folderze ───────────────────────────────────────────────
rng = np.random.default_rng(seed=42)

# Jesli zadany folder nie istnieje, stwórz
spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"USE CATALOG {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS default")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.default.{VOLUME}")

for t in range(1, T + 1):
    df = make_dataset(t, rng)
    path = f"{VOLUME_PATH}/dataset{t}.csv"
    # pandas pisze w csv
    df.to_csv(path, index=False)
    print(f"dataset{t}.csv  →  {len(df)} rows, sigma={float(np.sqrt(t)):.3f}")

print("\nDone. All datasets written to", VOLUME_PATH)
