import os, sys, json, time, requests
from sql_cases import CASES, QUESTIONS, SCHEMA

def state_of(c): return {"schema": SCHEMA, "intent": c["intent"], "sql": c["sql"]}

def run_jev():
    key = os.environ["TYPESAFE_API_KEY"]; out = {}
    for c in CASES:
        t0 = time.time()
        r = requests.post("https://api.typesafe.ai/v1/systemone", headers={"Authorization": f"Bearer {key}"},
                          json={"state": state_of(c), "model": "jev-latest", "questions": QUESTIONS}, timeout=60)
        r.raise_for_status(); a = r.json()["answers"]
        out[c["id"]] = {"safe": a["safe"]["noul"], "correct": a["correct"]["noul"], "cost": a["cost"]["score"],
                        "kind": a["kind"]["choice"], "kind_p": a["kind"]["probabilities"], "ms": round((time.time()-t0)*1000)}
    return out

def run_laya(ckpt):
    import laya; ag = laya.load("convaiinnovations/laya", subfolder=None if ckpt == "english" else ckpt); out = {}
    for c in CASES:
        t0 = time.time(); a = ag.predict(state_of(c), QUESTIONS)["answers"]
        out[c["id"]] = {"safe": a["safe"]["noul"], "correct": a["correct"]["noul"], "cost": a["cost"]["score"],
                        "kind": a["kind"]["choice"], "kind_p": a["kind"]["probabilities"], "ms": round((time.time()-t0)*1000)}
    return out

def score(name, res):
    n = len(CASES); s = dict(safe=0, correct=0, cost=0, kind=0); brier = dict(safe=0, correct=0)
    for c in CASES:
        r = res[c["id"]]
        s["safe"] += (r["safe"] >= .5) == c["safe"]; s["correct"] += (r["correct"] >= .5) == c["correct"]
        s["cost"] += round(r["cost"]) == c["cost"]; s["kind"] += r["kind"] == c["kind"]
        brier["safe"] += (r["safe"] - c["safe"])**2; brier["correct"] += (r["correct"] - c["correct"])**2
    ms = sorted(r["ms"] for r in res.values())[n//2]
    print(f"{name:22} safe {s['safe']}/{n}  correct {s['correct']}/{n}  cost {s['cost']}/{n}  kind {s['kind']}/{n}  "
          f"brier safe={brier['safe']/n:.3f} correct={brier['correct']/n:.3f}  p50={ms}ms")

if __name__ == "__main__":
    which = sys.argv[1:] or ["jev", "typed-decisions", "english", "multilingual"]
    all_res = json.load(open("sql_results.json")) if os.path.exists("sql_results.json") else {}
    for w in which:
        all_res[w] = run_jev() if w == "jev" else run_laya(w)
        json.dump(all_res, open("sql_results.json", "w"), indent=1)
    print(f"cases={len(CASES)}")
    for w, res in all_res.items(): score(w, res)
    print("\nper case  (gt: safe/correct/cost/kind)  ->  " + "  |  ".join(all_res))
    for c in CASES:
        cells = [f"{r[c['id']]['safe']:.2f}/{r[c['id']]['correct']:.2f}/{r[c['id']]['cost']:.1f}/{r[c['id']]['kind'][:4]}" for r in all_res.values()]
        print(f"{c['id']:18} {int(c['safe'])}/{int(c['correct'])}/{c['cost']}/{c['kind'][:4]}  ->  " + "  |  ".join(cells))
