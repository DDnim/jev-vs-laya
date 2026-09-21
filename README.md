# Jev vs Laya: SQL review benchmark

A small, reproducible benchmark comparing two "System 1" decision models — models that
take a **state** plus typed **questions** and return calibrated probabilities instead of text:

- **Jev** (`jev-latest`) — TypeSafe AI, proprietary API. https://docs.typesafe.ai
- **Laya** — Convai Innovations, open weights (Apache 2.0), 3 checkpoints. https://huggingface.co/convaiinnovations/laya

Both accept the same wire format (`state` + `questions` of type `choice` / `score` / `noul`),
so the exact same payload is sent to each.

## Task 1: review 14 SQL statements

Each case is a PostgreSQL statement with an explicit **intent** and a shared **schema context**
(5 tables, row counts, indexes). Seven of them are correct; seven contain a deliberate bug:

| bug | case |
|---|---|
| join fan-out double-counting a SUM | `q02_fanout` |
| `= NULL` (always UNKNOWN) | `q03_null_eq` |
| `OR` deletes far more than intended | `q04_delete_all` |
| `UPDATE` without `WHERE` | `q06_update_no_where` |
| cross join instead of `NOT EXISTS` | `q08_cross_join` |
| `CREATE INDEX` without `CONCURRENTLY` on 80M rows | `q10_migration_lock` |
| `BETWEEN` on dates includes the next day | `q13_between_ts` |
| `OR 1=1 --` injection remnant | `q14_injection` |

Four questions are asked per statement:

| id | type | question |
|---|---|---|
| `safe` | noul | read-only and safe to run on production as-is? |
| `correct` | noul | implements the intent with no logical bug? |
| `cost` | score 0–2 | cheap / moderate / heavy |
| `kind` | choice | report / destructive / schema |

## Results (2026-09-21)

| model | safe | correct | cost | kind | Brier safe / correct | p50 latency |
|---|---|---|---|---|---|---|
| **Jev `jev-latest`** (API) | **14/14** | **13/14** | 9/14 | **14/14** | **0.035 / 0.109** | 524 ms |
| Laya `typed-decisions` | 7/14 | 6/14 | 4/14 | 12/14 | 0.257 / 0.269 | 975 ms (M-series CPU) |
| Laya `english` | 5/14 | 6/14 | 4/14 | 2/14 | 0.330 / 0.348 | 1209 ms |
| Laya `multilingual` | 8/14 | 6/14 | 4/14 | 7/14 | 0.420 / 0.458 | 630 ms |

Per-case probabilities are in [`sql_results.json`](sql_results.json).

Observations:

- Jev caught the fan-out (`correct`=0.46), `= NULL` (0.06), cross join (0.03), missing
  `CONCURRENTLY` (0.43) and the injection (0.02). Its only miss is the `BETWEEN` off-by-one
  (0.83 "correct"). `safe` is well calibrated: 0.01–0.03 for DELETE/UPDATE/DDL, 0.73–0.90 for
  plain SELECTs, 0.10 for the injection.
- Laya's `safe`/`correct` sit at 0.4–0.6 (`typed-decisions`) or are all ≥0.9 (`multilingual`
  gives `safe`=0.97 to a `DELETE` that wipes two years of orders). It is not reading the SQL;
  only the surface-level `kind` question works (12/14 with `typed-decisions`).
- Laya's published "beats Jev" numbers come from its own four synthetic workflows and do not
  transfer to code review.

## Task 2: route a task card to an LLM tier + reasoning effort

20 synthetic task cards (mixed zh / ja / en, modelled on a personal Kanban) each labelled with
the LLM tier that should run it (`flash` / `sonnet` / `opus` / `fable`) and the reasoning effort
(0 low – 3 xhigh). The state is just `{"card": "..."}`; the tier descriptions live in the
question's `criteria`.

| model | tier | effort exact | effort ±1 | p50 latency |
|---|---|---|---|---|
| **Jev `jev-latest`** (API) | **17/20** | **19/20** | 20/20 | 545 ms |
| Laya `typed-decisions` | 9/20 | 7/20 | 19/20 | 289 ms (CPU) |
| Laya `multilingual` | 5/20 | 7/20 | 16/20 | 66 ms |

Per-case answers are in [`routing_results.json`](routing_results.json).

- Jev's three tier misses are all one step off on borderline cards (a research card as `opus`
  instead of `sonnet`, a spec'd multi-file feature as `sonnet` instead of `opus`). Its effort
  scores land on 0.0 / 1.0 / 2.0 / 3.0 almost exactly.
- Laya `typed-decisions` never picks `flash` for "say test" / "你是哪个模型" style cards and
  compresses effort into 1.3–2.1 for everything. `multilingual` is faster but close to random
  on tier (4 options → 25 % chance, it gets 25 %).

## Run it

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export TYPESAFE_API_KEY=...          # https://console.typesafe.ai/keys
python3 sql_bench.py                 # task 1: jev + all 3 Laya checkpoints → sql_results.json
python3 routing_bench.py             # task 2: jev + typed-decisions + multilingual → routing_results.json
python3 sql_bench.py jev             # a single model
```

Laya checkpoints are downloaded from Hugging Face on first use (~1.2 GB for all three).

## Files

- `sql_cases.py` — schema, 14 cases with ground truth, the 4 questions
- `sql_bench.py` — runs Jev (HTTP) and Laya (local), prints the tables above
- `sql_results.json` — raw answers from this run
- `routing_cases.py` / `routing_bench.py` / `routing_results.json` — task 2
