from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.database import init_db
from app.services.dropzone_catalog import (
    DEFAULT_CATALOG_PATH,
    audit_catalog,
    import_historical_observations,
    load_catalog,
    sync_catalog_file,
)
from app.services.dropzone_matching import audit_dropzone_matches


def main() -> int:
    parser = argparse.ArgumentParser(description="Dropzone-Katalog prüfen, importieren und auditieren.")
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--apply", action="store_true", help="Validierten Katalog in die SQLite-Datenbank importieren.")
    parser.add_argument(
        "--observe-history",
        action="store_true",
        help="Stabile Boden-GNSS-Sequenzen als Nachweise speichern; Sprünge werden nicht zugeordnet.",
    )
    parser.add_argument("--audit", action="store_true", help="Aktuellen Datenbankbestand auditieren.")
    parser.add_argument(
        "--audit-matches",
        action="store_true",
        help="Automatische Treffer und lokal gesammelte Prüffälle auditieren.",
    )
    parser.add_argument("--json", action="store_true", help="Ergebnis als JSON ausgeben.")
    args = parser.parse_args()

    init_db()
    if args.audit_matches:
        result = audit_dropzone_matches()
        action = "matches_audited"
    elif args.apply:
        result = sync_catalog_file(args.catalog)
        action = "imported"
    elif args.audit:
        result = audit_catalog()
        action = "audited"
    else:
        catalog = load_catalog(args.catalog)
        result = {
            "catalog_version": catalog["catalog_version"],
            "dropzones": len(catalog["dropzones"]),
            "zones": len(catalog["zones"]),
            "operators": len(catalog["operators"]),
            "sources": len(catalog["sources"]),
            "valid": True,
        }
        action = "validated"

    if args.observe_history:
        historical_import = import_historical_observations()
        result = audit_catalog()
        result["historical_import"] = historical_import
        action = "imported_and_observed" if args.apply else "observed"

    payload = {"action": action, **result}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"Dropzone-Katalog: {action}")
        for key, value in payload.items():
            if key != "action":
                print(f"- {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
