import os, sys, json, time, requests
from routing_cases import CASES, QUESTIONS

def state_of(c): return {"card": c["card"]}

def run_jev():
    key = os.environ["TYPESAFE_API_KEY"]; out = {}
    for c in CASES:
        t0 = time.time()
        r = requests.post("https://api.typesafe.ai/v1/systemone", headers={"Authorization": f"Bearer {key}"},
                          json={"state": state_of(c), "model": "jev-latest", "questions": QUESTIONS}, timeout=60)
        r.raise_for_status(); a = r.json()["answers"]
        out[c["id"]] = {"model": a["model"]["choice"], "model_p": a["model"]["probabilities"], "effort": a["effort"]["score"], "ms": round((time.time()-t0)*1000)}
    return out

def run_laya(ckpt):
    import laya; ag = laya.load("convaiinnovations/laya", subfolder=None if ckpt == "english" else ckpt); out = {}
    for c in CASES:
        t0 = time.time(); a = ag.predict(state_of(c), QUESTIONS)["answers"]
        out[c["id"]] = {"model": a["model"]["choice"], "model_p": a["model"]["probabilities"], "effort": a["effort"]["score"], "ms": round((time.time()-t0)*1000)}
    return out

def score(name, res):
    n = len(CASES); m = e1 = e0 = 0
    for c in CASES:
        r = res[c["id"]]; m += r["model"] == c["model"]
        e0 += round(r["effort"]) == c["effort"]; e1 += abs(round(r["effort"]) - c["effort"]) <= 1
    ms = sorted(r["ms"] for r in res.values())[n//2]
    print(f"{name:18} model {m}/{n}  effort exact {e0}/{n}  effort ±1 {e1}/{n}  p50={ms}ms")

if __name__ == "__main__":
    which = sys.argv[1:] or ["jev", "typed-decisions", "multilingual"]
    all_res = json.load(open("routing_results.json")) if os.path.exists("routing_results.json") else {}
    for w in which:
        all_res[w] = run_jev() if w == "jev" else run_laya(w)
        json.dump(all_res, open("routing_results.json", "w"), indent=1, ensure_ascii=False)
    print(f"cases={len(CASES)}")
    for w, res in all_res.items(): score(w, res)
    print("\nper case (gt model/effort) -> " + " | ".join(all_res))
    for c in CASES:
        print(f"{c['id']} {c['model']:6}/{c['effort']} -> " + " | ".join(f"{r[c['id']]['model']:6}/{r[c['id']]['effort']:.1f}" for r in all_res.values()))
