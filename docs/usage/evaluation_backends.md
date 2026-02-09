# Evaluation Backends

This repo supports two evaluation backends for SWE-bench:

- `docker` (local): runs the official Docker-based evaluator via `python -m swebench.harness.run_evaluation`.
- `sbcli` (cloud): submits predictions via `sb-cli` and waits for an evaluation report.

The primary correctness invariant is that every evaluated run directory should contain a
parseable `results.json` artifact. We treat this as the end-to-end contract:

- predictions existed (`preds.json` / `all.preds.json`)
- evaluation ran to completion
- totals can be recomputed (`resolved`, `evaluated`)

## Which Backend To Use

- Use `sbcli` when you want a reliable, decoupled evaluator and you have `SWEBENCH_API_KEY`.
  This is typically the safest option on shared VPS machines where Docker runs can hang.
- Use `docker` when you want fully local evaluation (no cloud dependency), or when you are
  evaluating subsets that are not supported by `sb-cli`.

## Verified-Mini (`verified-mini`)

`verified-mini` is a community 50-instance subset (MariusHobbhahn) and is **not** an official
sb-cli subset.

To support `verified-mini` safely with `sb-cli`, we:

- map `verified-mini` to the sb-cli subset `swe-bench_verified`, and
- submit only the instance IDs present in the run's predictions using `--instance_ids`.

Guardrails prevent accidental full-benchmark submissions:

- refuse if `0` instance IDs can be extracted from predictions
- refuse if more than `60` instance IDs are present

## Backfilling Missing Evaluations

Use `scripts/evaluate_missing.py` to backfill runs that are missing a valid `results.json`:

```bash
source .venv/bin/activate

# Verified-Mini (50) backfill using sb-cli (default)
python scripts/evaluate_missing.py --subset verified-mini --backend sbcli --workers 1 --timeout 900

# Full Verified (500) backfill using sb-cli
python scripts/evaluate_missing.py --subset verified --backend sbcli --workers 1 --timeout 900

# Force local Docker evaluation (not recommended on shared VPS)
python scripts/evaluate_missing.py --subset verified-mini --backend docker --workers 1 --timeout 900
```

`evaluate_missing.py` treats `results.json` as a contract; a file that exists but cannot be parsed
or reports `n_evaluated == 0` is treated as missing and will be backfilled.

