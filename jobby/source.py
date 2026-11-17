"""Read listings from the main DuckDB database, always read-only.

The main database is an analysis store owned by the scrapers: Jobby never
writes to it. Only current SCD versions are read, so one listing is never
processed twice (see jobby-plan.md item 7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import duckdb

COLUMNS = (
    "job_id",
    "job_board",
    "url",
    "company",
    "title",
    "city",
    "description",
    "technologies",
    "seniority",
    "work_mode",
    "employment_types",
    "salary_month",
    "salary_hour",
    "currency",
    "salary_type",
    "date_posted",
)

POLISH_DIACRITICS = set("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ")


class SourceError(RuntimeError):
    """Raised when the source database cannot be read."""


@dataclass(frozen=True)
class Listing:
    job_id: str
    job_board: str | None
    url: str | None
    company: str | None
    title: str | None
    city: str | None
    description: str
    technologies: list[str] = field(default_factory=list)
    seniority: str | None = None
    work_mode: str | None = None
    employment_types: list[str] = field(default_factory=list)
    salary_month: int | None = None
    salary_hour: int | None = None
    currency: str | None = None
    salary_type: str | None = None
    date_posted: date | None = None

    @property
    def language(self) -> str:
        """Rough language guess: Polish diacritics mean a Polish description."""
        return "pl" if any(ch in self.description for ch in POLISH_DIACRITICS) else "en"


def connect(path: str | Path) -> duckdb.DuckDBPyConnection:
    db_path = Path(path)
    if not db_path.is_file():
        raise SourceError(f"source database not found: {db_path}")
    try:
        return duckdb.connect(str(db_path), read_only=True)
    except duckdb.Error as exc:
        raise SourceError(f"cannot open {db_path} read-only: {exc}") from exc


def _to_list(value: Any) -> list[str]:
    if not value:
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def fetch_listings(
    path: str | Path,
    *,
    table: str = "dim_job_listings",
    limit: int | None = None,
) -> list[Listing]:
    """Return current listings, newest first, with a non-empty description."""
    con = connect(path)
    try:
        sql = (
            f"select {', '.join(COLUMNS)} from {table} "
            "where is_current and description is not null and length(trim(description)) > 0 "
            "order by date_posted desc nulls last, scraped_at desc"
        )
        if limit:
            sql += f" limit {int(limit)}"
        rows = con.execute(sql).fetchall()
    except duckdb.Error as exc:
        raise SourceError(f"query failed on {path}: {exc}") from exc
    finally:
        con.close()

    return [
        Listing(
            job_id=str(row[0]),
            job_board=row[1],
            url=row[2],
            company=row[3],
            title=row[4],
            city=row[5],
            description=str(row[6]),
            technologies=_to_list(row[7]),
            seniority=row[8],
            work_mode=row[9],
            employment_types=_to_list(row[10]),
            salary_month=row[11],
            salary_hour=row[12],
            currency=row[13],
            salary_type=row[14],
            date_posted=row[15],
        )
        for row in rows
    ]
