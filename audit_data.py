#!/usr/bin/env python3
"""Audita estrutura e cobertura do dataset sem assumir sua qualidade."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).parent
SOURCE_META = json.loads((HERE / "data_source.json").read_text(encoding="utf-8"))
DATA = Path(SOURCE_META["download_path"])
PARQUET = DATA / "daily_prices.parquet"
ITEMS = DATA / "items.csv"
OUT = HERE / "auditoria_dados.json"


def main() -> None:
    parquet_file = pq.ParquetFile(PARQUET)
    schema = [(field.name, str(field.type)) for field in parquet_file.schema_arrow]
    prices = pd.read_parquet(PARQUET)
    items = pd.read_csv(ITEMS, low_memory=False)

    date = pd.to_datetime(prices["date"], errors="coerce")
    numeric = ["price_median", "price_mean", "price_min", "price_max", "volume"]
    report = {
        "source": str(PARQUET),
        "rows": int(len(prices)),
        "columns": list(prices.columns),
        "schema": schema,
        "date_min": str(date.min().date()),
        "date_max": str(date.max().date()),
        "distinct_items": int(prices["market_hash_name"].nunique()),
        "duplicate_item_date_rows": int(prices.duplicated(["market_hash_name", "date"]).sum()),
        "missing": {c: int(prices[c].isna().sum()) for c in prices.columns},
        "non_positive": {
            c: int((pd.to_numeric(prices[c], errors="coerce") <= 0).sum()) for c in numeric
        },
        "items_catalog_rows": int(len(items)),
        "items_catalog_columns": list(items.columns),
        "memory_bytes_loaded": int(prices.memory_usage(deep=True).sum()),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
