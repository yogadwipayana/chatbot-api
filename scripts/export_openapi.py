"""Ekspor spesifikasi OpenAPI yang dihasilkan FastAPI.

`api.yaml` ditulis tangan dan mencakup endpoint yang belum diimplementasikan,
jadi ia TIDAK dihasilkan dari script ini. Gunanya untuk membandingkan: saat
sebuah endpoint selesai dikerjakan, keluaran script ini adalah rujukan untuk
memperbarui bagian `implemented` di api.yaml.

    python -m scripts.export_openapi                 # cetak YAML ke stdout
    python -m scripts.export_openapi --json          # cetak JSON
    python -m scripts.export_openapi -o live.yaml    # simpan ke berkas

Perbedaan untuk endpoint `implemented` ditangkap otomatis oleh
`tests/api/test_openapi_contract.py`; script ini hanya membantu melihat
persisnya beda di mana.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.main import create_app


def main() -> int:
    parser = argparse.ArgumentParser(description="Ekspor OpenAPI dari aplikasi")
    parser.add_argument("--json", action="store_true", help="Keluarkan JSON, bukan YAML")
    parser.add_argument("-o", "--output", type=Path, help="Simpan ke berkas")
    args = parser.parse_args()

    spec = create_app().openapi()
    teks = (
        json.dumps(spec, indent=2, ensure_ascii=False)
        if args.json
        else yaml.safe_dump(spec, sort_keys=False, allow_unicode=True, width=100)
    )

    if args.output:
        args.output.write_text(teks, encoding="utf-8")
        print(f"tersimpan: {args.output}", file=sys.stderr)
    else:
        print(teks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
