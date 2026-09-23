"""Evaluate a fine-tuned Laya on (a) the 14 handwritten cases and (b) the 240-case synthetic hold-out."""
import sys, json, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import laya
from sql_cases import CASES, QUESTIONS

MODEL = sys.argv[1]
ag = laya.Agent(MODEL) if os.path.exists(MODEL) else laya.load("convaiinnovations/laya", subfolder=MODEL)
KINDS = list(QUESTIONS["kind"]["criteria"])

def predict(state):
    a = ag.predict(state, QUESTIONS)["answers"]
    return a["safe"]["noul"], a["correct"]["noul"], a["cost"]["score"], a["kind"]["choice"]

# (a) 手書き 14
s = dict(safe=0, correct=0, cost=0, kind=0); br = dict(safe=0.0, correct=0.0); lat = []
for c in CASES:
    t0 = time.time(); sa, co, cs, ki = predict({"schema": __import__("sql_cases").SCHEMA, "intent": c["intent"], "sql": c["sql"]}); lat.append(time.time()-t0)
    s["safe"] += (sa >= .5) == c["safe"]; s["correct"] += (co >= .5) == c["correct"]
    s["cost"] += round(cs) == c["cost"]; s["kind"] += ki == c["kind"]
    br["safe"] += (sa - c["safe"])**2; br["correct"] += (co - c["correct"])**2
n = len(CASES)
print(f"[handwritten 14] safe {s['safe']}/{n}  correct {s['correct']}/{n}  cost {s['cost']}/{n}  kind {s['kind']}/{n}  "
      f"brier {br['safe']/n:.3f}/{br['correct']/n:.3f}  p50={sorted(lat)[n//2]*1000:.0f}ms")

# (b) 合成 hold-out
test = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_examples.json")))
t = dict(safe=0, correct=0, cost=0, kind=0)
for e in test:
    sa, co, cs, ki = predict({"schema": e["schema"], "intent": e["intent"], "sql": e["sql"]})
    t["safe"] += (sa >= .5) == e["safe"]; t["correct"] += (co >= .5) == e["correct"]
    t["cost"] += round(cs) == e["cost"]; t["kind"] += ki == e["kind"]
m = len(test)
print(f"[synthetic hold-out {m}] safe {t['safe']/m:.3f}  correct {t['correct']/m:.3f}  cost {t['cost']/m:.3f}  kind {t['kind']/m:.3f}")
