from __future__ import annotations

import argparse
import html as html_lib
import json
import re
import shutil
import sys
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Optional

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_process.clean import preprocess_documents


TEXT_SUFFIXES = {".htm", ".html", ".asp", ".php", ".jsp", ".psp"}
JSONL_SKIP_TABLES = {
    "sqlite_sequence",
    "chunks_fts",
    "chunks_fts_config",
    "chunks_fts_data",
    "chunks_fts_docsize",
    "chunks_fts_idx",
}


@dataclass(frozen=True)
class PipelineConfig:
    input_dir: Path
    output_dir: Path
    max_chars: int = 900
    overlap_chars: int = 120
    copy_raw: bool = True
    copy_texts: bool = True
    build_cleaned: bool = True
    build_sqlite: bool = True
    verify: bool = True


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def normalize_rel_path(rel_path: Optional[str]) -> Optional[Path]:
    if not rel_path:
        return None
    return Path(rel_path.replace("\\", "/"))


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_html_text(html_text: str) -> str:
    try:
        from bs4 import BeautifulSoup  # type: ignore
    except Exception:
        BeautifulSoup = None  # type: ignore

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html_text, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html_text)
        text = re.sub(r"(?s)<[^>]+>", "\n", text)

    text = html_lib.unescape(text)
    text = re.sub(r"[ \t\f\v]+\n", "\n", text)
    text = re.sub(r"\n[ \t\f\v]+", "\n", text)
    return collapse_whitespace(text)


def extract_pdf_text(pdf_path: Path) -> str:
    try:
        from PyPDF2 import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyPDF2 is required to extract PDF text") from exc

    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return collapse_whitespace("\n\n".join(pages))


def extract_text_from_raw(raw_path: Path) -> str:
    suffix = raw_path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return extract_html_text(raw_path.read_text(encoding="utf-8", errors="ignore"))
    if suffix == ".pdf":
        return extract_pdf_text(raw_path)
    return raw_path.read_text(encoding="utf-8", errors="ignore")


def source_sqlite_path(input_dir: Path) -> Optional[Path]:
    candidates = [
        input_dir / "sist_kb.sqlite",
        input_dir / "sist" / "sist_kb.sqlite",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def source_jsonl_dir(input_dir: Path) -> Optional[Path]:
    candidates = [input_dir / "jsonl", input_dir / "sist" / "jsonl"]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_documents(input_dir: Path) -> list[dict]:
    jsonl_dir = source_jsonl_dir(input_dir)
    if jsonl_dir:
        docs_path = jsonl_dir / "documents.jsonl"
        if docs_path.exists():
            return list(read_jsonl(docs_path))

    sqlite_path = source_sqlite_path(input_dir)
    if sqlite_path:
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT id, run_id, url, canonical_url, title, host, category, language,
                       content_type, status_code, fetched_at, source_published_at,
                       valid_from, valid_until, validity_note, raw_path, text_path,
                       sha256, depth, parent_url, text_chars
                FROM documents
                ORDER BY id
                """
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()
    raise FileNotFoundError("No documents source found")


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def materialize_files(config: PipelineConfig, documents: list[dict]) -> None:
    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()

    if config.copy_raw:
        copy_tree(input_dir / "raw", output_dir / "raw")

    if config.copy_texts:
        copy_tree(input_dir / "texts", output_dir / "texts")

    for doc in documents:
        raw_rel = normalize_rel_path(doc.get("raw_path"))
        text_rel = normalize_rel_path(doc.get("text_path"))

        raw_src = input_dir / raw_rel if raw_rel else None
        text_dst = output_dir / text_rel if text_rel else None

        if text_dst and text_dst.exists():
            continue
        if not text_dst or not raw_src or not raw_src.exists():
            continue

        text_dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            text_dst.write_text(extract_text_from_raw(raw_src), encoding="utf-8")
        except Exception:
            # If a single document cannot be converted, keep the pipeline moving.
            pass


def export_jsonl_sources(config: PipelineConfig, documents: list[dict]) -> None:
    input_dir = config.input_dir.resolve()
    output_jsonl = config.output_dir.resolve() / "jsonl"
    output_jsonl.mkdir(parents=True, exist_ok=True)

    src_jsonl = source_jsonl_dir(input_dir)
    if src_jsonl:
        for path in src_jsonl.glob("*.jsonl"):
            shutil.copy2(path, output_jsonl / path.name)
        return

    sqlite_path = source_sqlite_path(input_dir)
    if not sqlite_path:
        raise FileNotFoundError("No jsonl directory or sqlite database found")

    conn = sqlite3.connect(str(sqlite_path))
    conn.row_factory = sqlite3.Row
    try:
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            if row[0] not in JSONL_SKIP_TABLES
        ]
        for table in tables:
            rows = conn.execute(f"SELECT * FROM {table} ORDER BY id" if table != "crawl_runs" else f"SELECT * FROM {table} ORDER BY id").fetchall()
            write_jsonl(output_jsonl / f"{table}.jsonl", (dict(row) for row in rows))
    finally:
        conn.close()


def load_table_rows(jsonl_dir: Path, table: str) -> list[dict]:
    path = jsonl_dir / f"{table}.jsonl"
    if not path.exists():
        return []
    return list(read_jsonl(path))


def create_schema_from_template(conn: sqlite3.Connection, template_sqlite: Path) -> None:
    tmpl = sqlite3.connect(str(template_sqlite))
    try:
        statements = tmpl.execute(
            """
            SELECT sql
            FROM sqlite_master
            WHERE sql IS NOT NULL
              AND name NOT LIKE 'sqlite_%'
              AND name NOT LIKE 'chunks_fts_%'
            ORDER BY rowid
            """
        ).fetchall()
        for (sql,) in statements:
            if sql:
                conn.execute(sql)
    finally:
        tmpl.close()


def insert_rows(
    conn: sqlite3.Connection,
    table: str,
    rows: list[dict],
    *,
    reset_seq: bool = True,
) -> None:
    if not rows:
        return

    columns = [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
    insert_cols = [col for col in columns if col != "rowid"]
    sample = rows[0]
    usable_cols = [col for col in insert_cols if col in sample or any(col in row for row in rows)]
    if not usable_cols:
        return

    placeholders = ", ".join(["?"] * len(usable_cols))
    col_sql = ", ".join(usable_cols)
    sql = f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})"
    values = [[row.get(col) for col in usable_cols] for row in rows]
    conn.executemany(sql, values)

    if reset_seq and "id" in columns:
        max_id = max((int(row["id"]) for row in rows if row.get("id") is not None), default=None)
        if max_id is not None:
            conn.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
            conn.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES(?, ?)",
                (table, max_id),
            )


def build_sqlite_database(config: PipelineConfig) -> Path:
    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()
    output_db = output_dir / "sist_kb.sqlite"
    template_db = source_sqlite_path(input_dir)
    if not template_db:
        raise FileNotFoundError("Cannot find template sqlite database")

    jsonl_dir = output_dir / "jsonl"
    if not jsonl_dir.exists():
        raise FileNotFoundError("jsonl directory not found in output directory")

    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(str(output_db))
    try:
        conn.execute("PRAGMA foreign_keys = OFF")
        create_schema_from_template(conn, template_db)
        conn.commit()

        ordered_tables = [
            "crawl_runs",
            "documents",
            "chunks",
            "entities",
            "facts",
            "events",
            "leadership_roles",
            "program_sources",
            "program_requirements",
            "facilities",
            "staff_members",
            "faculty_members",
            "courses",
            "contacts",
            "openinfo_documents",
            "clinic_services",
            "dining_venues",
            "dorm_buildings",
            "transport_routes",
            "administrative_units",
        ]

        for table in ordered_tables:
            rows = load_table_rows(jsonl_dir, table)
            if rows:
                insert_rows(conn, table, rows)

        conn.commit()
        if "chunks_fts" in {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'") if row[0]}:
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        conn.commit()
        conn.execute("PRAGMA foreign_keys = ON")
        conn.commit()
    finally:
        conn.close()

    return output_db


def count_rows(sqlite_path: Path, table: str) -> int:
    conn = sqlite3.connect(str(sqlite_path))
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def verify_build(config: PipelineConfig, output_db: Path) -> None:
    source_db = source_sqlite_path(config.input_dir)
    if not source_db:
        return

    conn = sqlite3.connect(str(source_db))
    try:
        tables = [
            row[0]
            for row in conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
            if row[0] not in JSONL_SKIP_TABLES
        ]
        mismatches: list[str] = []
        for table in tables:
            src = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            dst = count_rows(output_db, table)
            if src != dst:
                mismatches.append(f"{table}: source={src}, output={dst}")
        if mismatches:
            raise RuntimeError("Row count mismatch:\n" + "\n".join(mismatches))
    finally:
        conn.close()


def build_pipeline(config: PipelineConfig) -> dict:
    input_dir = config.input_dir.resolve()
    output_dir = config.output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("output_dir must differ from input_dir to avoid modifying source data")

    output_dir.mkdir(parents=True, exist_ok=True)
    documents = load_documents(input_dir)

    materialize_files(config, documents)

    cleaned_stats = None
    if config.build_cleaned:
        cleaned_output = output_dir / "processed"
        docs, chunks = preprocess_documents(
            input_dir=input_dir,
            output_dir=cleaned_output,
            max_chars=config.max_chars,
            overlap_chars=config.overlap_chars,
        )
        cleaned_stats = {"documents": len(docs), "chunks": len(chunks)}

    export_jsonl_sources(config, documents)

    output_db = None
    if config.build_sqlite:
        output_db = build_sqlite_database(config)
        if config.verify:
            verify_build(config, output_db)

    result = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "documents": len(documents),
        "cleaned": cleaned_stats,
        "sqlite": str(output_db) if output_db else None,
    }
    (output_dir / "pipeline_stats.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild the SIST knowledge base pipeline.")
    parser.add_argument("--input-dir", type=Path, default=Path("sist"), help="Source SIST directory")
    parser.add_argument("--output-dir", type=Path, default=Path("rebuilt_sist"), help="Output directory")
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per cleaned chunk")
    parser.add_argument("--overlap-chars", type=int, default=120, help="Overlap in cleaned chunks")
    parser.add_argument("--no-copy-raw", action="store_true", help="Do not copy raw files")
    parser.add_argument("--no-copy-texts", action="store_true", help="Do not copy text files")
    parser.add_argument("--skip-cleaned", action="store_true", help="Skip cleaned documents/chunks generation")
    parser.add_argument("--skip-sqlite", action="store_true", help="Skip rebuilding sqlite")
    parser.add_argument("--skip-verify", action="store_true", help="Skip row-count verification")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    config = PipelineConfig(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
        copy_raw=not args.no_copy_raw,
        copy_texts=not args.no_copy_texts,
        build_cleaned=not args.skip_cleaned,
        build_sqlite=not args.skip_sqlite,
        verify=not args.skip_verify,
    )
    result = build_pipeline(config)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
