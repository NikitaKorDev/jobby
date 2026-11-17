# jobby

Reads job listings collected by
[job-market-analysis](https://github.com/NikitaKorDev/job-market-analysis),
ranks them against my resume with a local Ollama model, and prepares a
tailored resume for each listing I choose to apply to.

## Dependency on job-market-analysis

Jobby does not scrape anything. It opens the scrapers' DuckDB database
read-only (table `dim_job_listings`, current rows only). The path is set in
`config/jobby.yaml`:

```yaml
source:
  mode: duckdb                  # or "fixture" for fixtures/sample.duckdb
  path: ../job_listings.duckdb  # relative to this folder
```

The default path assumes this repo is checked out inside the
job-market-analysis folder, which gitignores it. To keep it anywhere else,
point `source.path` at the database.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put your resume at `resume/<name>.docx` and set `resume.source_path` in
`config/jobby.yaml`. Resumes are gitignored.
Ollama must be running with the model set in `ollama.model`.

To rebuild the demo database, run `python fixtures/build_fixture.py`.
