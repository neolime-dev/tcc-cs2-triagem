#!/usr/bin/env python3
"""Baixa o dataset público selecionado para o protótipo.

Fonte: Kaggle, devynpruden/cs2-skin-price-history-2013-2026
Licença declarada no metadado: Apache 2.0
"""
from __future__ import annotations

import json
from pathlib import Path

import kagglehub

HANDLE = "devynpruden/cs2-skin-price-history-2013-2026"
OUT = Path(__file__).parent / "data_source.json"


def main() -> None:
    path = Path(kagglehub.dataset_download(HANDLE))
    files = [
        {"name": str(p.relative_to(path)), "bytes": p.stat().st_size}
        for p in path.rglob("*")
        if p.is_file()
    ]
    metadata = {
        "handle": HANDLE,
        "download_path": str(path),
        "license_declared": "Apache 2.0",
        "files": files,
    }
    OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
