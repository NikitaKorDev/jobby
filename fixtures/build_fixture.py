"""Build the small demo database Jobby runs against until the real one is upgraded.

Real listings, real descriptions, real urls: 12 offers taken deterministically
from the scrapers' backup (job_data_raw.json) and written in the Phase-3 schema
from the root plan.md. Re-run this script to rebuild the fixture byte-for-byte.

    .venv/bin/python jobby/fixtures/build_fixture.py

The current job_listings.duckdb does not store url/description, so Jobby refuses
to read it; this fixture exists so the pipeline can be exercised end to end.
"""

from __future__ import annotations

import json
import random
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import duckdb

FIXTURE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = FIXTURE_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
RAW_FILE = REPO_ROOT / "job_data_raw.json"
OUT_FILE = FIXTURE_DIR / "sample.duckdb"

BOARDS = ("olx", "justjoin", "nofluff", "bulldogjobs")
PER_BOARD = 3
SEED = 20261115

SCHEMA = """
CREATE OR REPLACE TABLE dim_job_listings (
    surrogate_key    VARCHAR PRIMARY KEY,
    job_id           VARCHAR NOT NULL,
    job_board        VARCHAR,
    url              VARCHAR,
    company          VARCHAR,
    title            VARCHAR,
    city             VARCHAR,
    description      VARCHAR,
    technologies     VARCHAR[],
    salary_month     INTEGER,
    salary_hour      INTEGER,
    currency         VARCHAR,
    salary_type      VARCHAR,
    employment_types VARCHAR[],
    seniority        VARCHAR,
    work_mode        VARCHAR,
    application_url  VARCHAR,
    benefits         VARCHAR[],
    date_posted      DATE,
    scraped_at       TIMESTAMP,
    hash_diff        VARCHAR,
    effective_from   DATE NOT NULL,
    effective_to     DATE,
    is_current       BOOLEAN NOT NULL
);
"""


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _date(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = _text(value)
    return [part.strip() for part in text.split(",") if part.strip()] if text else []


def _row(listing: dict[str, Any], effective_from: date) -> tuple[Any, ...]:
    offer_id = _text(listing.get("offer_id")) or _text(listing.get("url"))
    board = _text(listing.get("board")) or "unknown"
    job_id = offer_id or ""
    technologies = _list(listing.get("technologies_list")) or _list(listing.get("technologies"))
    hash_diff = f"fixture:{board}:{job_id}"
    return (
        f"fixture-{board}-{job_id}"[:64],
        job_id,
        board,
        _text(listing.get("url")),
        _text(listing.get("company")),
        _text(listing.get("title")),
        _text(listing.get("city")),
        _text(listing.get("description")),
        technologies,
        _int(listing.get("salary-month")),
        _int(listing.get("salary-hour")),
        _text(listing.get("currency")),
        _text(listing.get("salary_type")),
        _list(listing.get("employment_types")),
        _text(listing.get("seniority")),
        _text(listing.get("work_mode")),
        _text(listing.get("application_url")),
        _list(listing.get("benefits")),
        _date(listing.get("date")),
        _timestamp(listing.get("scraped_at")),
        hash_diff,
        effective_from,
        date(9999, 12, 31),
        True,
    )


def select_listings(raw: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    picked: list[dict[str, Any]] = []
    for board in BOARDS:
        candidates = [
            item
            for item in raw.get(board, [])
            if item.get("description") and item.get("url") and item.get("title")
        ]
        candidates.sort(key=lambda item: str(item.get("offer_id") or item.get("url")))
        if len(candidates) <= PER_BOARD:
            picked.extend(candidates)
            continue
        chosen = rng.sample(candidates, PER_BOARD)
        # Guarantee one untagged listing so the pre-filter path is exercised.
        if board == BOARDS[0] and not any(
            not _list(item.get("technologies_list")) for item in chosen
        ):
            untagged = [
                item for item in candidates if not _list(item.get("technologies_list"))
            ]
            if untagged:
                chosen[-1] = untagged[0]
        picked.extend(chosen)
    picked.sort(key=lambda item: (str(item.get("board")), str(item.get("offer_id"))))
    return picked


def main() -> int:
    if not RAW_FILE.is_file():
        print(f"raw scraper backup not found: {RAW_FILE}", file=sys.stderr)
        return 1
    with RAW_FILE.open(encoding="utf-8") as handle:
        raw = json.load(handle)

    picked = select_listings(raw)
    effective_from = date.today()
    rows = [_row(item, effective_from) for item in picked]

    if OUT_FILE.exists():
        OUT_FILE.unlink()
    con = duckdb.connect(str(OUT_FILE))
    con.execute(SCHEMA)
    con.executemany(
        "insert into dim_job_listings values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    con.close()

    print(f"wrote {OUT_FILE} ({OUT_FILE.stat().st_size / 1024:.1f} KiB), {len(rows)} listings")
    for item in picked:
        desc = _text(item.get("description")) or ""
        lang = "pl" if any(ch in desc for ch in "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ") else "en"
        techs = _list(item.get("technologies_list"))
        print(
            f"  {item['board']:<11} {item['offer_id'][:48]:<48} "
            f"{len(desc):>6} chars  {lang}  tags={len(techs)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
