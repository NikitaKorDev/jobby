"""Configuration loading for Jobby.

All tunables live in config/jobby.yaml. Relative paths resolve against the
project folder (the one that contains config/). Nothing is hard-coded here
beyond the defaults required to read that file.
"""

from __future__ import annotations

import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_CONFIG_PATH = PROJECT_DIR / "config" / "jobby.yaml"


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid."""


@dataclass(frozen=True)
class SourceConfig:
    mode: str
    path: Path
    table: str


@dataclass(frozen=True)
class StorageConfig:
    sqlite: Path


@dataclass(frozen=True)
class ResumeConfig:
    source_path: Path
    criteria: Path
    tag_aliases: Path


@dataclass(frozen=True)
class OllamaConfig:
    url: str
    model: str
    num_ctx: int
    num_predict: int
    temperature: float
    timeout_s: float


@dataclass(frozen=True)
class BatchConfig:
    max_per_run: int | None
    newest_first: bool


@dataclass(frozen=True)
class PrefilterConfig:
    include_untagged: bool


@dataclass(frozen=True)
class OnlineCheckConfig:
    timeout_s: float
    user_agent: str


@dataclass(frozen=True)
class ApiConfig:
    host: str
    port: int
    open_browser: bool


@dataclass(frozen=True)
class PathsConfig:
    resumes_dir: Path
    bench_dir: Path


@dataclass(frozen=True)
class Config:
    source: SourceConfig
    storage: StorageConfig
    resume: ResumeConfig
    ollama: OllamaConfig
    batch: BatchConfig
    prefilter: PrefilterConfig
    online_check: OnlineCheckConfig
    api: ApiConfig
    paths: PathsConfig
    config_path: Path
    project_dir: Path


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name) or {}
    if not isinstance(value, dict):
        raise ConfigError(f"config section '{name}' must be a mapping")
    return value


def _path(value: Any, root: Path, what: str) -> Path:
    if not value:
        raise ConfigError(f"missing path for '{what}'")
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path | None = None) -> Config:
    config_path = Path(path).expanduser().resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")
    root = config_path.parent.parent

    source = _section(raw, "source")
    storage = _section(raw, "storage")
    resume = _section(raw, "resume")
    ollama = _section(raw, "ollama")
    batch = _section(raw, "batch")
    prefilter = _section(raw, "prefilter")
    online = _section(raw, "online_check")
    api = _section(raw, "api")
    paths = _section(raw, "paths")

    mode = str(source.get("mode", "fixture")).strip().lower()
    if mode not in {"fixture", "duckdb"}:
        raise ConfigError(f"source.mode must be 'fixture' or 'duckdb', got {mode!r}")

    max_per_run = batch.get("max_per_run")
    if max_per_run is not None:
        max_per_run = int(max_per_run)
        if max_per_run <= 0:
            raise ConfigError("batch.max_per_run must be a positive integer or null")

    cfg = Config(
        source=SourceConfig(
            mode=mode,
            path=_path(source.get("path"), root, "source.path"),
            table=str(source.get("table", "dim_job_listings")),
        ),
        storage=StorageConfig(
            sqlite=_path(storage.get("sqlite"), root, "storage.sqlite"),
        ),
        resume=ResumeConfig(
            source_path=_path(resume.get("source_path"), root, "resume.source_path"),
            criteria=_path(resume.get("criteria", "config/criteria.yaml"), root, "resume.criteria"),
            tag_aliases=_path(resume.get("tag_aliases", "config/tag_aliases.yaml"), root, "resume.tag_aliases"),
        ),
        ollama=OllamaConfig(
            url=str(ollama.get("url", "auto")),
            model=str(ollama.get("model", "qwen3:4b")),
            num_ctx=int(ollama.get("num_ctx", 8192)),
            num_predict=int(ollama.get("num_predict", 320)),
            temperature=float(ollama.get("temperature", 0)),
            timeout_s=float(ollama.get("timeout_s", 900)),
        ),
        batch=BatchConfig(
            max_per_run=max_per_run,
            newest_first=bool(batch.get("newest_first", True)),
        ),
        prefilter=PrefilterConfig(
            include_untagged=bool(prefilter.get("include_untagged", True)),
        ),
        online_check=OnlineCheckConfig(
            timeout_s=float(online.get("timeout_s", 4)),
            user_agent=str(online.get("user_agent", "Mozilla/5.0 (compatible; Jobby/0.1)")),
        ),
        api=ApiConfig(
            host=str(api.get("host", "127.0.0.1")),
            port=int(api.get("port", 8000)),
            open_browser=bool(api.get("open_browser", True)),
        ),
        paths=PathsConfig(
            resumes_dir=_path(paths.get("resumes_dir", "resumes"), root, "paths.resumes_dir"),
            bench_dir=_path(paths.get("bench_dir", "bench"), root, "paths.bench_dir"),
        ),
        config_path=config_path,
        project_dir=root,
    )
    return cfg


def read_criteria(cfg: Config) -> str:
    """Render criteria.yaml into prompt-ready text."""
    if not cfg.resume.criteria.is_file():
        raise ConfigError(f"criteria file not found: {cfg.resume.criteria}")
    raw = yaml.safe_load(cfg.resume.criteria.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{cfg.resume.criteria}: top level must be a mapping")
    lines: list[str] = []
    must = raw.get("must_have") or []
    if isinstance(must, str):
        must = [must]
    if must:
        lines.append("MUST-HAVE RULES (violating any of them makes the listing not worth applying to):")
        lines.extend(f"- {str(item).strip()}" for item in must)
    notes = str(raw.get("notes") or "").strip()
    if notes:
        lines.append("")
        lines.append("ADDITIONAL GUIDANCE:")
        lines.append(notes)
    if not lines:
        raise ConfigError(f"{cfg.resume.criteria}: define at least one must_have rule or notes")
    return "\n".join(lines)


def _default_gateway() -> str | None:
    """Return the WSL default gateway (the Windows host) if it can be read."""
    try:
        for line in Path("/proc/net/route").read_text().splitlines()[1:]:
            fields = line.split()
            if len(fields) > 2 and fields[1] == "00000000":
                hex_ip = fields[2]
                return socket.inet_ntoa(bytes.fromhex(hex_ip)[::-1])
    except OSError:
        return None
    return None


def ollama_candidates(cfg: Config) -> list[str]:
    """Ordered base URLs to try when ollama.url is 'auto'."""
    explicit = cfg.ollama.url.strip().rstrip("/")
    if explicit and explicit.lower() != "auto":
        return [explicit]
    candidates: list[str] = []
    env_url = os.environ.get("OLLAMA_URL") or os.environ.get("OLLAMA_HOST")
    if env_url:
        if "://" not in env_url:
            env_url = f"http://{env_url}"
        candidates.append(env_url.rstrip("/"))
    candidates.append("http://localhost:11434")
    gateway = _default_gateway()
    if gateway:
        candidates.append(f"http://{gateway}:11434")
    seen: list[str] = []
    for item in candidates:
        if item not in seen:
            seen.append(item)
    return seen


def resolve_ollama_url(cfg: Config, *, timeout_s: float = 2.0) -> str:
    """Find a reachable Ollama base URL; raise ConfigError if none answers."""
    import httpx

    candidates = ollama_candidates(cfg)
    for base in candidates:
        try:
            response = httpx.get(f"{base}/api/tags", timeout=timeout_s)
            if response.status_code == 200:
                return base
        except httpx.HTTPError:
            continue
    raise ConfigError(
        "Ollama is not reachable. Tried: "
        + ", ".join(candidates)
        + ". Start Ollama (Windows host for WSL) and/or set ollama.url in config/jobby.yaml."
    )
