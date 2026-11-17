"""Model benchmark for the determinator (jobby-plan.md item 8).

Runs candidate Ollama models over real listings from the source database and
reports, per model: time per listing, throughput, JSON validity and the actual
verdicts. Nothing is decided automatically: the report is meant to be read.

    cd jobby && ../.venv/bin/python -m jobby.bench
    cd jobby && ../.venv/bin/python -m jobby.bench --models qwen3:4b,qwen3:8b
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import prompts
from .config import Config, load_config, read_criteria, resolve_ollama_url
from .ollama_client import OllamaError, chat, parse_json_object
from .resume import resume_text
from .source import Listing, fetch_listings

DEFAULT_MODELS = ("qwen3:4b", "qwen3:8b")
CHARS_PER_TOKEN = 3.5
TRANSCRIPT_KEYS = ("worth_it", "chance", "reasoning", "missing_required")


@dataclass
class Run:
    model: str
    index: int
    job_id: str
    title: str
    language: str
    wall_s: float
    prompt_tokens: int
    eval_tokens: int
    tokens_per_second: float
    truncated: bool
    raw: str
    error: str | None
    parsed: dict[str, Any] | None

    @property
    def valid_json(self) -> bool:
        return self.parsed is not None

    @property
    def valid_schema(self) -> bool:
        if self.parsed is None:
            return False
        worth = str(self.parsed.get("worth_it", "")).strip().lower()
        chance = self.parsed.get("chance")
        missing = self.parsed.get("missing_required")
        reasoning = self.parsed.get("reasoning")
        chance_ok = isinstance(chance, (int, float)) or (
            isinstance(chance, str) and chance.strip().rstrip("%").isdigit()
        )
        return (
            worth in {"yes", "no"}
            and chance_ok
            and isinstance(missing, list)
            and all(isinstance(item, str) for item in missing)
            and isinstance(reasoning, str)
            and bool(reasoning.strip())
        )


def _bucket(seniority: str | None) -> str:
    value = (seniority or "").strip().lower()
    if value in {"junior", "intern", "trainee", "internship"}:
        return "junior"
    if value in {"mid", "regular"}:
        return "mid"
    if value:
        return "senior"
    return "unknown"


def _balanced_by_language(items: list[Listing], count: int) -> list[Listing]:
    """Pick ``count`` listings alternating PL/EN, newest first inside each."""
    by_language = {"pl": [item for item in items if item.language == "pl"],
                   "en": [item for item in items if item.language == "en"]}
    order = sorted(by_language, key=lambda lang: len(by_language[lang]), reverse=True)
    picked: list[Listing] = []
    cursors = {lang: 0 for lang in order}
    while len(picked) < count:
        progressed = False
        for lang in order:
            pool = by_language[lang]
            if cursors[lang] < len(pool) and len(picked) < count:
                picked.append(pool[cursors[lang]])
                cursors[lang] += 1
                progressed = True
        if not progressed:
            break
    return picked


def _matches_tags(listing: Listing, technologies: list[str]) -> bool:
    """Case-insensitive overlap between listing tags and the resume tags."""
    known = {item.strip().lower() for item in technologies if item.strip()}
    if not known:
        return True
    tags = {item.strip().lower() for item in listing.technologies if item.strip()}
    return bool(tags & known)


def pick_listings(
    listings: list[Listing], count: int, *, pool: str = "tag", technologies: list[str] | None = None
) -> list[Listing]:
    """Sample newest-300 listings, stratified by seniority and balanced by language.

    Seniority is the main thing the criteria judge, so the sample deliberately
    includes junior, mid and senior roles instead of only the newest ones.
    With ``pool='tag'`` only listings whose tags overlap the resume tags are
    eligible, mirroring the pipeline's pre-filter.
    """
    candidates = listings
    if pool == "tag" and technologies:
        candidates = [item for item in listings if _matches_tags(item, technologies)]
    window = candidates[:300]
    buckets: dict[str, list[Listing]] = {"junior": [], "mid": [], "senior": [], "unknown": []}
    for listing in window:
        buckets[_bucket(listing.seniority)].append(listing)

    shares = [("junior", 3), ("mid", 3), ("senior", 4)]
    picked: list[Listing] = []
    for name, share in shares:
        pool = buckets[name]
        if not pool:
            continue
        picked.extend(_balanced_by_language(pool, share))
    if len(picked) < count:
        rest = [item for item in window if item not in picked]
        picked.extend(_balanced_by_language(rest, count - len(picked)))
    picked.sort(key=lambda item: (item.date_posted is None, -(item.date_posted.toordinal() if item.date_posted else 0)))
    return picked[:count]


def fit_description(
    *, prompt_without_description: str, description: str, num_ctx: int, num_predict: int
) -> tuple[str, bool]:
    """Trim the description so the prompt fits the explicit context window."""
    budget = int((num_ctx - num_predict - 100) * CHARS_PER_TOKEN) - len(prompt_without_description)
    if len(description) <= budget:
        return description, False
    if budget <= 200:
        return "", True
    cut = description[:budget]
    return cut + "\n[... description truncated to fit the context window ...]", True


def extract_tags(
    *,
    base_url: str,
    model: str,
    cfg: Config,
    resume: str,
    cache_file: Path,
    force: bool = False,
) -> list[str]:
    """Extract the candidate's technologies once; reused by every model run."""
    if cache_file.is_file() and not force:
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        return [str(item) for item in cached.get("technologies", [])]

    result = chat(
        base_url,
        model,
        system=prompts.SYSTEM,
        prompt=prompts.tag_prompt(resume=resume),
        num_ctx=cfg.ollama.num_ctx,
        num_predict=cfg.ollama.num_predict,
        temperature=cfg.ollama.temperature,
        timeout_s=cfg.ollama.timeout_s,
    )
    parsed, error = parse_json_object(result.text)
    if parsed is None or not isinstance(parsed.get("technologies"), list):
        raise OllamaError(f"tag extraction failed ({error}): {result.text[:300]}")
    technologies = [str(item).strip() for item in parsed["technologies"] if str(item).strip()]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps(
            {
                "model": model,
                "wall_s": round(result.wall_s, 2),
                "technologies": technologies,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return technologies


def run_model(
    *,
    base_url: str,
    model: str,
    cfg: Config,
    listings: list[Listing],
    resume: str,
    technologies: list[str],
    criteria: str,
    out_dir: Path,
) -> list[Run]:
    print(f"\n=== {model} ({len(listings)} listings) ===", flush=True)
    # Warm-up: load the model and prime nothing else, excluded from the stats.
    chat(
        base_url,
        model,
        system=prompts.SYSTEM,
        prompt='Reply with exactly {"ok": true}',
        num_ctx=cfg.ollama.num_ctx,
        num_predict=32,
        temperature=0,
        timeout_s=cfg.ollama.timeout_s,
    )

    runs: list[Run] = []
    for index, listing in enumerate(listings, start=1):
        prefix = prompts.evaluation_prompt(
            resume=resume,
            technologies=technologies,
            criteria=criteria,
            listing=listing,
            description="",
        )
        description, truncated = fit_description(
            prompt_without_description=prefix,
            description=listing.description,
            num_ctx=cfg.ollama.num_ctx,
            num_predict=cfg.ollama.num_predict,
        )
        prompt = prompts.evaluation_prompt(
            resume=resume,
            technologies=technologies,
            criteria=criteria,
            listing=listing,
            description=description,
        )
        try:
            result = chat(
                base_url,
                model,
                system=prompts.SYSTEM,
                prompt=prompt,
                num_ctx=cfg.ollama.num_ctx,
                num_predict=cfg.ollama.num_predict,
                temperature=cfg.ollama.temperature,
                timeout_s=cfg.ollama.timeout_s,
            )
            parsed, error = parse_json_object(result.text)
            run = Run(
                model=model,
                index=index,
                job_id=listing.job_id,
                title=listing.title or "?",
                language=listing.language,
                wall_s=result.wall_s,
                prompt_tokens=result.prompt_tokens,
                eval_tokens=result.eval_tokens,
                tokens_per_second=result.tokens_per_second,
                truncated=truncated,
                raw=result.text,
                error=error,
                parsed=parsed,
            )
        except OllamaError as exc:
            run = Run(
                model=model,
                index=index,
                job_id=listing.job_id,
                title=listing.title or "?",
                language=listing.language,
                wall_s=0.0,
                prompt_tokens=0,
                eval_tokens=0,
                tokens_per_second=0.0,
                truncated=truncated,
                raw="",
                error=str(exc),
                parsed=None,
            )
        runs.append(run)
        with (out_dir / "runs.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(run), ensure_ascii=False) + "\n")
        verdict = run.parsed.get("worth_it") if run.parsed else "-"
        print(
            f"  {index:>2}/{len(listings)} {run.wall_s:6.1f}s  "
            f"{run.tokens_per_second:5.1f} tok/s  "
            f"{'OK ' if run.valid_schema else 'BAD'}  worth_it={verdict}  "
            f"{run.title[:48]}",
            flush=True,
        )
    return runs


def _fmt(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}"


def write_report(
    *,
    path: Path,
    cfg: Config,
    models: list[str],
    listings: list[Listing],
    runs: list[Run],
    technologies: list[str],
    tag_model: str,
    base_url: str,
    pool_label: str,
) -> None:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = [
        "# Determinator benchmark",
        "",
        f"- Run: {now}",
        f"- Ollama: {base_url}",
        f"- Models: {', '.join(models)}",
        f"- num_ctx: {cfg.ollama.num_ctx}, num_predict: {cfg.ollama.num_predict}, "
        f"temperature: {cfg.ollama.temperature}, thinking: off",
        f"- Listings: {len(listings)} real current listings from `{cfg.source.path}` ({pool_label})",
        f"- Tags extracted by `{tag_model}` from the source resume: {len(technologies)} technologies",
        f"- Criteria: `{cfg.resume.criteria.name}` (unconfirmed, see jobby-plan.md item 12)",
        "",
        "## Speed and output format",
        "",
        "| model | median s | mean s | p90 s | tok/s (eval) | prompt tok | valid JSON | valid schema |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model in models:
        subset = [run for run in runs if run.model == model and run.error is None]
        walls = [run.wall_s for run in subset]
        speeds = [run.tokens_per_second for run in subset if run.tokens_per_second > 0]
        prompts_tok = [run.prompt_tokens for run in subset]
        valid_json = sum(1 for run in subset if run.valid_json)
        valid_schema = sum(1 for run in subset if run.valid_schema)
        if not walls:
            lines.append(f"| {model} | - | - | - | - | - | 0/{len(listings)} | 0/{len(listings)} |")
            continue
        p90 = sorted(walls)[max(0, int(len(walls) * 0.9) - 1)]
        lines.append(
            f"| {model} | {_fmt(statistics.median(walls))} | {_fmt(statistics.fmean(walls))} | "
            f"{_fmt(p90)} | {_fmt(statistics.fmean(speeds) if speeds else 0)} | "
            f"{int(statistics.fmean(prompts_tok))} | "
            f"{valid_json}/{len(listings)} | {valid_schema}/{len(listings)} |"
        )

    lines += ["", "## Verdicts side by side", ""]
    header = "| # | lang | seniority | title | " + " | ".join(models) + " | agree |"
    lines += [header, "|" + "---|" * (len(models) + 5)]
    for index, listing in enumerate(listings, start=1):
        row = [
            str(index),
            listing.language,
            listing.seniority or "-",
            (listing.title or "?")[:52].replace("|", "/"),
        ]
        verdicts: list[str] = []
        for model in models:
            run = next(
                (r for r in runs if r.model == model and r.index == index and r.parsed),
                None,
            )
            if run is None:
                verdicts.append("-")
                row.append("-")
                continue
            worth = str(run.parsed.get("worth_it", "?")).strip().lower()
            chance = run.parsed.get("chance")
            verdicts.append(worth)
            row.append(f"{worth}/{chance}")
        agree = "yes" if len(set(verdicts)) == 1 and verdicts[0] != "-" else "no"
        row.append(agree)
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Reasoning samples", ""]
    for run in runs:
        if not run.parsed:
            continue
        if run.index not in {1, 3, 6}:
            continue
        missing = run.parsed.get("missing_required") or []
        lines.append(
            f"**{run.model} · #{run.index} {run.title[:60]}** — "
            f"worth_it={run.parsed.get('worth_it')}, chance={run.parsed.get('chance')}, "
            f"missing={missing}"
        )
        lines.append("")
        lines.append(f"> {str(run.parsed.get('reasoning', '')).strip()}")
        lines.append("")

    failures = [run for run in runs if not run.valid_schema]
    if failures:
        lines += ["## Failures", ""]
        for run in failures:
            lines.append(f"- {run.model} #{run.index}: {run.error or 'schema mismatch'}")
            lines.append(f"  ```\n  {(run.raw or '')[:500]}\n  ```")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark determinator models on real listings.")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--listings", type=int, default=10)
    parser.add_argument("--tag-model", default="qwen3:8b")
    parser.add_argument("--num-ctx", type=int, default=None)
    parser.add_argument("--num-predict", type=int, default=400)
    parser.add_argument(
        "--pool",
        choices=("tag", "all"),
        default="tag",
        help="tag = only listings whose tags overlap the resume tags (pipeline pre-filter); all = raw newest",
    )
    parser.add_argument("--out", default=None, help="results folder (default: config paths.bench_dir)")
    parser.add_argument("--force-tags", action="store_true", help="re-extract resume tags")
    args = parser.parse_args(argv)

    from dataclasses import replace

    cfg = load_config()
    cfg = replace(
        cfg,
        ollama=replace(
            cfg.ollama,
            num_ctx=args.num_ctx or cfg.ollama.num_ctx,
            num_predict=args.num_predict,
        ),
    )
    base_url = resolve_ollama_url(cfg)
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    out_dir = Path(args.out).resolve() if args.out else cfg.paths.bench_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_file = out_dir / "runs.jsonl"
    if runs_file.exists():
        runs_file.unlink()

    print(f"Ollama: {base_url}")
    print(f"Output: {out_dir}")

    resume = resume_text(cfg.resume.source_path)
    technologies = extract_tags(
        base_url=base_url,
        model=args.tag_model,
        cfg=cfg,
        resume=resume,
        cache_file=out_dir / "resume_tags.json",
        force=args.force_tags,
    )
    print(f"Resume: {len(resume)} chars, {len(technologies)} tags: {', '.join(technologies)}")

    criteria = read_criteria(cfg)
    listings = fetch_listings(cfg.source.path, table=cfg.source.table)
    picked = pick_listings(
        listings, args.listings, pool=args.pool, technologies=technologies
    )
    pool_label = "tag pre-filter" if args.pool == "tag" else "raw newest, no pre-filter"
    print(f"Listings: {len(picked)} picked ({pool_label}) from {len(listings)} current rows")
    for listing in picked:
        print(
            f"  {listing.language}  {str(listing.date_posted)}  "
            f"{len(listing.description):>6}ch  {listing.seniority or '-':<10} "
            f"{(listing.title or '?')[:60]}"
        )

    runs: list[Run] = []
    for model in models:
        try:
            runs.extend(
                run_model(
                    base_url=base_url,
                    model=model,
                    cfg=cfg,
                    listings=picked,
                    resume=resume,
                    technologies=technologies,
                    criteria=criteria,
                    out_dir=out_dir,
                )
            )
        except OllamaError as exc:
            print(f"!! {model}: {exc}", file=sys.stderr, flush=True)

    report = out_dir / "report.md"
    write_report(
        path=report,
        cfg=cfg,
        models=models,
        listings=picked,
        runs=runs,
        technologies=technologies,
        tag_model=args.tag_model,
        base_url=base_url,
        pool_label=pool_label,
    )
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
