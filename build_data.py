"""Build SQL-review training data from Gretel synthetic_text_to_sql: originals (correct) + mutations (buggy)."""
import re, json, random, collections
from datasets import load_dataset
random.seed(0)
N_TARGET = int(__import__("sys").argv[1]) if len(__import__("sys").argv) > 1 else 1600

ds = load_dataset("gretelai/synthetic_text_to_sql", split="train")
rows = [r for r in ds if len(r["sql"]) < 700 and len(r["sql_context"]) < 1500]
random.shuffle(rows)
by_type = collections.defaultdict(list)
for r in rows: by_type[r["sql_task_type"]].append(r)
# 書き込み/DDL を多めに拾う
pick = by_type["data manipulation"][:int(N_TARGET*0.2)] + by_type["data definition"][:int(N_TARGET*0.1)] + by_type["analytics and reporting"][:int(N_TARGET*0.6)] + by_type["data retrieval"][:int(N_TARGET*0.1)]
random.shuffle(pick)

def kind_of(sql):
    m = re.match(r"\s*(\w+)", sql)
    if not m: return None
    w = m.group(1).upper()
    if w in ("SELECT", "WITH"): return "report"
    if w in ("INSERT", "UPDATE", "DELETE", "MERGE"): return "destructive"
    if w in ("CREATE", "ALTER", "DROP", "TRUNCATE"): return "schema"
    return None

COMPLEX_COST = {"basic SQL": 0, "aggregation": 1, "single join": 1, "subqueries": 1, "window functions": 2, "multiple_joins": 2, "set operations": 2, "CTEs": 2}

def base_labels(r, sql):
    k = kind_of(sql)
    cost = 2 if k != "report" else COMPLEX_COST.get(r["sql_complexity"], 1)
    return dict(kind=k, safe=(k == "report"), cost=cost)

# --- 変異: (name, applicable?, apply) → (sql, label_overrides)
def m_null(s):
    if re.search(r"\bIS NOT NULL\b", s, re.I): return re.sub(r"\bIS NOT NULL\b", "<> NULL", s, count=1, flags=re.I), {}
    if re.search(r"\bIS NULL\b", s, re.I): return re.sub(r"\bIS NULL\b", "= NULL", s, count=1, flags=re.I), {}
def m_andor(s):
    m = re.search(r"\bWHERE\b.*?\b(AND|OR)\b", s, re.I | re.S)
    if not m: return None
    i, j = m.span(1); return s[:i] + ("OR" if m.group(1).upper() == "AND" else "AND") + s[j:], {}
def m_nowhere(s):
    if kind_of(s) != "destructive" or not re.search(r"\bWHERE\b", s, re.I): return None
    return re.sub(r"\s+WHERE\b.*?(;|$)", r";", s, count=1, flags=re.I | re.S), {"cost": 2}
def m_cmp(s):
    for a, b in ((">=", ">"), ("<=", "<"), ("<", "<="), (">", ">=")):
        if re.search(r"\bWHERE\b.*" + re.escape(a) + r"(?!=)", s, re.I | re.S) and a in s:
            return s.replace(a, b, 1), {}
def m_agg(s):
    for a, b in (("SUM(", "COUNT("), ("AVG(", "SUM("), ("MAX(", "MIN("), ("MIN(", "MAX("), ("COUNT(DISTINCT ", "COUNT(")):
        if a in s.upper():
            i = s.upper().index(a); return s[:i] + b + s[i+len(a):], {}
def m_order(s):
    if not re.search(r"\bLIMIT\b", s, re.I) or not re.search(r"\bORDER BY\b", s, re.I): return None
    if re.search(r"\bDESC\b", s, re.I): return re.sub(r"\bDESC\b", "ASC", s, count=1, flags=re.I), {}
    return re.sub(r"(ORDER BY\s+[\w.]+)", r"\1 DESC", s, count=1, flags=re.I), {}
def m_inject(s):
    if kind_of(s) != "report" or not re.search(r"\bWHERE\b", s, re.I): return None
    return re.sub(r"(\bWHERE\b[^;]*?)(\s+(GROUP|ORDER|LIMIT)\b|;|$)", r"\1 OR 1=1 --\2", s, count=1, flags=re.I | re.S), {"safe": False, "cost": 2}
def m_cross(s):
    m = re.search(r"\bJOIN\s+([\w.]+)(\s+(?:AS\s+)?\w+)?\s+ON\s+[\w.]+\s*=\s*[\w.]+", s, re.I)
    if not m: return None
    return s[:m.start()] + ", " + m.group(1) + (m.group(2) or "") + s[m.end():], {"cost": 2}
def m_groupby(s):
    m = re.search(r"\bGROUP BY\s+([\w.]+)\s*,\s*([\w.]+)", s, re.I)
    if not m: return None
    return s[:m.start()] + "GROUP BY " + m.group(1) + s[m.end():], {}
def m_concurrently(s):
    if re.search(r"^\s*CREATE INDEX\b", s, re.I) and "CONCURRENTLY" not in s.upper():
        return None
    return None
MUTS = [m_null, m_andor, m_nowhere, m_cmp, m_agg, m_order, m_inject, m_cross, m_groupby]

out = []
for r in pick:
    sql = r["sql"].strip(); k = kind_of(sql)
    if not k: continue
    schema = "; ".join(x.strip() for x in re.split(r";\s*", r["sql_context"]) if x.strip().upper().startswith("CREATE"))[:1200]
    lab = base_labels(r, sql)
    out.append(dict(schema=schema, intent=r["sql_prompt"], sql=sql, correct=True, **lab, mut="none"))
    cands = [(m.__name__, m(sql)) for m in MUTS]
    cands = [(n, c) for n, c in cands if c and c[0] != sql]
    if random.random() < 0.15 and len(rows) > 1:   # intent 差し替え
        other = random.choice(rows)["sql_prompt"]
        if other != r["sql_prompt"]: cands.append(("m_intent", (sql, {"intent": other})))
    if cands:
        w = [0.3 if n == 'm_inject' else 1.0 for n, _ in cands]
        n, (msql, ov) = random.choices(cands, weights=w)[0]
        ex = dict(schema=schema, intent=ov.pop("intent", r["sql_prompt"]), sql=msql, correct=False, **lab, mut=n); ex.update(ov)
        out.append(ex)
random.shuffle(out)
json.dump(out, open("train_examples.json", "w"), indent=1)
print(len(out), collections.Counter(e["mut"] for e in out)); print(collections.Counter(e["kind"] for e in out), collections.Counter(e["safe"] for e in out), collections.Counter(e["cost"] for e in out))
for e in [x for x in out if x["mut"] != "none"][:6]: print("\n", e["mut"], "|", e["intent"][:60], "\n ", e["sql"][:200])
