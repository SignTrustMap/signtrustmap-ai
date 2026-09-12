import argparse
import json
import os
import sys
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).resolve().parent.parent
CATEGORIES_FILE = BASE_DIR / "configs" / "categories.json"
SIGNS_FILE = BASE_DIR / "configs" / "sign_metadata.json"
SQL_OUTPUT_FILE = BASE_DIR / "scripts" / "seed_traffic_signs.sql"


def load_data():
    if not CATEGORIES_FILE.exists():
        logger.error(f"Categories file not found: {CATEGORIES_FILE}")
        sys.exit(1)
    if not SIGNS_FILE.exists():
        logger.error(f"Sign metadata file not found: {SIGNS_FILE}")
        sys.exit(1)

    with open(CATEGORIES_FILE, "r", encoding="utf-8") as f:
        categories = json.load(f)

    with open(SIGNS_FILE, "r", encoding="utf-8") as f:
        signs = json.load(f)

    return categories, signs


def generate_sql(categories, signs) -> str:
    sql_statements = [
        "-- SignTrustMap Database Seed Script",
        "-- Auto-generated for Categories and 100 Traffic Signs (QCVN 41:2019)",
        "BEGIN;",
        "",
        "-- 1. Ensure Categories exist (creating PROHIBITORY, INFORMATION, etc. if not present)",
    ]

    for cat in categories:
        cat_id = cat["id"]
        code = cat["code"].replace("'", "''")
        name_vi = cat["name_vi"].replace("'", "''")
        name_en = cat["name_en"].replace("'", "''")
        desc = cat["description"].replace("'", "''")

        sql_statements.append(
            f"INSERT INTO categories (id, code, name_vi, name_en, description) "
            f"VALUES ({cat_id}, '{code}', '{name_vi}', '{name_en}', '{desc}') "
            f"ON CONFLICT (code) DO UPDATE SET "
            f"name_vi = EXCLUDED.name_vi, name_en = EXCLUDED.name_en, description = EXCLUDED.description;"
        )

    sql_statements.extend([
        "",
        "-- 2. Insert/Upsert 100 AI Traffic Sign Classes",
    ])

    for k in sorted(signs.keys(), key=lambda x: int(x)):
        sign = signs[k]
        cid = sign["class_id"]
        class_name = sign["class_name"].replace("'", "''")
        category = sign["category"].replace("'", "''")
        name_vi = sign["name_vi"].replace("'", "''")
        name_en = sign["name_en"].replace("'", "''")
        std = sign["standard_code"].replace("'", "''")

        sql_statements.append(
            f"INSERT INTO signs (class_id, sign_code, category_code, name_vi, name_en, standard_code) "
            f"VALUES ({cid}, '{class_name}', '{category}', '{name_vi}', '{name_en}', '{std}') "
            f"ON CONFLICT (sign_code) DO UPDATE SET "
            f"class_id = EXCLUDED.class_id, category_code = EXCLUDED.category_code, "
            f"name_vi = EXCLUDED.name_vi, name_en = EXCLUDED.name_en, standard_code = EXCLUDED.standard_code;"
        )

    sql_statements.extend([
        "",
        "COMMIT;",
        "",
    ])

    return "\n".join(sql_statements)


def main():
    parser = argparse.ArgumentParser(description="Seed categories and 100 AI traffic signs.")
    parser.add_argument("--sql-only", action="store_true", help="Only output SQL script to file")
    parser.add_argument("--print-sql", action="store_true", help="Print SQL to stdout")
    args = parser.parse_args()

    logger.info("Loading categories and sign metadata...")
    categories, signs = load_data()
    logger.info(f"Loaded {len(categories)} categories and {len(signs)} sign metadata items.")

    sql_content = generate_sql(categories, signs)

    SQL_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SQL_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(sql_content)
    logger.info(f"Generated SQL seed file at: {SQL_OUTPUT_FILE}")

    if args.print_sql:
        sys.stdout.write(sql_content)


if __name__ == "__main__":
    main()
