from __future__ import annotations

import argparse
import html as html_lib
import json
import os
import re
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator, Optional


NAVIGATION_LINES = {
    "English",
    "中文",
    "学校首页",
    "学校主页",
    "首页",
    "首 页",
    "导航",
    "返回",
    "当前位置：",
    "更多+",
    "更多",
    "Home",
    "About US",
    "About",
    "Research",
    "People",
    "Academics",
    "News",
    "Events",
    "Copyright © 上海科技大学 版权所有",
    "沪公网安备 31011502006855号",
    "地址：上海市浦东新区华夏中路393号 邮编：201210",
    "学校官微",
    "学院官微",
}


@dataclass
class DocumentRecord:
    id: int
    url: str
    title: str
    category: Optional[str]
    language: Optional[str]
    source_path: Optional[str]
    content_type: Optional[str]
    text_chars_raw: int
    text_chars_clean: int
    num_chunks: int


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: int
    chunk_index: int
    title: str
    url: str
    category: Optional[str]
    language: Optional[str]
    text: str
    char_count: int
    source_path: Optional[str]


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


def normalize_path(base_dir: Path, rel_path: Optional[str]) -> Optional[Path]:
    if not rel_path:
        return None
    return (base_dir / rel_path.replace("\\", "/")).resolve()


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html_tags(text: str) -> str:
    text = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", "\n", text)
    text = html_lib.unescape(text)
    return collapse_whitespace(text)


def is_boilerplate_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    if stripped in NAVIGATION_LINES:
        return True
    if stripped.startswith("Copyright ©"):
        return True
    if stripped.startswith("沪公网安备"):
        return True
    if stripped.startswith("地址："):
        return True
    if stripped.startswith("当前位置："):
        return True
    if stripped.endswith("更多+") or stripped == "更多":
        return True
    if len(stripped) <= 1:
        return True
    if re.fullmatch(r"[|·•—\-_=+*/\\]+", stripped):
        return True
    return False


def dedupe_consecutive(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    prev = None
    for line in lines:
        if line != prev:
            out.append(line)
        prev = line
    return out


def clean_text(text: str) -> str:
    text = strip_html_tags(text)
    lines = []
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if is_boilerplate_line(line):
            continue
        if line and len(line) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", line):
            continue
        lines.append(line)
    lines = dedupe_consecutive(lines)
    return collapse_whitespace("\n".join(lines))


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？!?；;\n])\s*", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 900, overlap_chars: int = 120) -> list[str]:
    text = collapse_whitespace(text)
    if not text:
        return []

    segments = [seg.strip() for seg in re.split(r"\n{2,}", text) if seg.strip()]
    units: list[str] = []
    for seg in segments:
        if len(seg) <= max_chars:
            units.append(seg)
            continue
        units.extend(sentence_split(seg))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush() -> None:
        nonlocal current, current_len
        if not current:
            return
        chunks.append("\n\n".join(current).strip())
        current = []
        current_len = 0

    for unit in units:
        if not unit:
            continue
        projected = current_len + len(unit) + (2 if current else 0)
        if current and projected > max_chars:
            flush()
        current.append(unit)
        current_len += len(unit) + (2 if len(current) > 1 else 0)
        if current_len >= max_chars:
            flush()

    flush()

    if overlap_chars <= 0 or len(chunks) <= 1:
        return chunks

    overlapped: list[str] = []
    prev_tail = ""
    for chunk in chunks:
        if prev_tail:
            merged = prev_tail + "\n\n" + chunk
            overlapped.append(merged)
        else:
            overlapped.append(chunk)
        prev_tail = chunk[-overlap_chars:]
    return overlapped


def read_document_text(doc: dict, base_dir: Path) -> tuple[str, Optional[Path]]:
    text_path = normalize_path(base_dir, doc.get("text_path"))
    if text_path and text_path.exists():
        return text_path.read_text(encoding="utf-8", errors="ignore"), text_path

    raw_path = normalize_path(base_dir, doc.get("raw_path"))
    if raw_path and raw_path.exists():
        suffix = raw_path.suffix.lower()
        if suffix in {".htm", ".html", ".asp", ".php", ".jsp", ".psp"}:
            return raw_path.read_text(encoding="utf-8", errors="ignore"), raw_path
        if suffix == ".txt":
            return raw_path.read_text(encoding="utf-8", errors="ignore"), raw_path
        if suffix == ".pdf":
            try:
                from PyPDF2 import PdfReader  # type: ignore
            except Exception:
                PdfReader = None  # type: ignore
            if PdfReader is not None:
                reader = PdfReader(str(raw_path))
                pages = []
                for page in reader.pages:
                    pages.append(page.extract_text() or "")
                return "\n".join(pages), raw_path
    return "", None


def load_chunks(jsonl_path: Optional[Path]) -> list[dict]:
    if jsonl_path and jsonl_path.exists():
        return list(read_jsonl(jsonl_path))
    return []


def load_documents(jsonl_path: Optional[Path], sqlite_path: Optional[Path]) -> list[dict]:
    if jsonl_path and jsonl_path.exists():
        return list(read_jsonl(jsonl_path))
    # if sqlite_path and sqlite_path.exists():
    #     conn = sqlite3.connect(str(sqlite_path))
    #     conn.row_factory = sqlite3.Row
    #     try:
    #         rows = conn.execute(
    #             """
    #             SELECT id, url, canonical_url, title, host, category, language,
    #                    content_type, raw_path, text_path, text_chars
    #             FROM documents
    #             ORDER BY id
    #             """
    #         ).fetchall()
    #         return [dict(row) for row in rows]
    #     finally:
    #         conn.close()
    raise FileNotFoundError("No documents source found")


def preprocess_documents(
    input_dir: Path,
    output_dir: Path,
    max_chars: int = 900,
    overlap_chars: int = 120,
) -> tuple[list[DocumentRecord], list[ChunkRecord]]:
    input_dir = input_dir.resolve()
    docs_path = input_dir / "merged_documents.jsonl"
    chunks_path = input_dir / "merged_chunks.jsonl"
    sqlite_path = None
    # if not docs_path.exists():
    #     docs_path = input_dir / "jsonl" / "documents.jsonl"
    #     sqlite_path = input_dir / "sist_kb.sqlite"
    #     if not sqlite_path.exists():
    #         sqlite_path = input_dir / "sist" / "sist_kb.sqlite"
    docs = load_documents(docs_path, sqlite_path)
    merged_chunks = load_chunks(chunks_path)
    chunks_by_document_id: dict[int, list[dict]] = {}
    for chunk in merged_chunks:
        document_id = chunk.get("document_id")
        if document_id is None:
            continue
        chunks_by_document_id.setdefault(int(document_id), []).append(chunk)

    documents: list[DocumentRecord] = []
    chunks: list[ChunkRecord] = []

    for doc in docs:
        title = doc.get("title") or ""
        url = doc.get("url") or doc.get("canonical_url") or ""
        category = doc.get("category")
        language = doc.get("language")
        source_path = doc.get("text_path") or doc.get("raw_path") or doc.get("source_path")
        source_path_str = None
        if source_path:
            source_path_str = str(source_path)

        document_id = int(doc["id"])
        raw_chunks = chunks_by_document_id.get(document_id, [])

        if raw_chunks:
            cleaned_rows: list[tuple[dict, str]] = []
            for raw_chunk in raw_chunks:
                cleaned_chunk = clean_text(raw_chunk.get("text", ""))
                if cleaned_chunk:
                    cleaned_rows.append((raw_chunk, cleaned_chunk))

            if not cleaned_rows:
                continue

            documents.append(
                DocumentRecord(
                    id=document_id,
                    url=url,
                    title=title,
                    category=category,
                    language=language,
                    source_path=source_path_str,
                    content_type=doc.get("content_type"),
                    text_chars_raw=int(doc.get("text_chars") or 0),
                    text_chars_clean=sum(len(cleaned_chunk) for _, cleaned_chunk in cleaned_rows),
                    num_chunks=len(cleaned_rows),
                )
            )

            for raw_chunk, cleaned_chunk in cleaned_rows:
                chunk_index = int(raw_chunk.get("chunk_index") or 0)
                chunk_id = raw_chunk.get("chunk_id") or f"{document_id}-{chunk_index}"
                chunks.append(
                    ChunkRecord(
                        chunk_id=str(chunk_id),
                        document_id=document_id,
                        chunk_index=chunk_index,
                        title=title,
                        url=url,
                        category=category,
                        language=language,
                        text=f"{title}\n{url}\n\n{cleaned_chunk}".strip(),
                        char_count=len(cleaned_chunk),
                        source_path=source_path_str,
                    )
                )
            continue

        raw_text, source_path_file = read_document_text(doc, input_dir)
        if not raw_text:
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        if source_path_file:
            try:
                source_path_str = str(source_path_file.relative_to(input_dir).as_posix())
            except ValueError:
                source_path_str = str(source_path_file.as_posix())

        chunk_texts = chunk_text(cleaned, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunk_texts:
            continue

        documents.append(
            DocumentRecord(
                id=document_id,
                url=url,
                title=title,
                category=category,
                language=language,
                source_path=source_path_str,
                content_type=doc.get("content_type"),
                text_chars_raw=int(doc.get("text_chars") or len(raw_text)),
                text_chars_clean=len(cleaned),
                num_chunks=len(chunk_texts),
            )
        )

        for idx, chunk in enumerate(chunk_texts):
            chunk = chunk.strip()
            if not chunk:
                continue
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{document_id}-{idx}",
                    document_id=document_id,
                    chunk_index=idx,
                    title=title,
                    url=url,
                    category=category,
                    language=language,
                    text=f"{title}\n{url}\n\n{chunk}".strip(),
                    char_count=len(chunk),
                    source_path=source_path_str,
                )
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "documents_cleaned.jsonl", (asdict(row) for row in documents))
    write_jsonl(output_dir / "chunks_cleaned.jsonl", (asdict(row) for row in chunks))

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "max_chars": max_chars,
        "overlap_chars": overlap_chars,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
    }
    (output_dir / "preprocess_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return documents, chunks


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess SIST knowledge base data.")
    parser.add_argument("--input-dir", type=Path, default=Path("/data/qingyuyang/dlproject/data/processed"), help="Input SIST data directory")
    parser.add_argument("--output-dir", type=Path, default=Path("/data/qingyuyang/dlproject/data/processed"), help="Output directory")
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per chunk")
    parser.add_argument("--overlap-chars", type=int, default=120, help="Chunk overlap in characters")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    docs, chunks = preprocess_documents(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    print(
        json.dumps(
            {
                "documents": len(docs),
                "chunks": len(chunks),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
