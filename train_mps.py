"""Fine-tune Laya on the SQL-review data with RLCD (single-device mps/cpu port of the upstream 2xT4 DDP notebook)."""
import os, sys, time, json, random
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from laya.agent import _fix_tokenizer_config
from laya.common import build_model, build_sequence, render_options, proper_reward, QTYPES

SUBFOLDER, OUT = sys.argv[1], sys.argv[2]
EPOCHS = int(os.environ.get("EPOCHS", 2)); MICRO = int(os.environ.get("MICRO", 4)); ACCUM = int(os.environ.get("ACCUM", 8))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", 0)); STEP_TIMING = os.environ.get("STEP_TIMING")
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

root = snapshot_download("convaiinnovations/laya", allow_patterns=[f"{SUBFOLDER}/*"] if SUBFOLDER != "english" else None)
mdir = os.path.join(root, SUBFOLDER) if SUBFOLDER != "english" else root
_fix_tokenizer_config(mdir)
tok = AutoTokenizer.from_pretrained(os.path.join(mdir, "tokenizer"))
cfg = json.load(open(os.path.join(mdir, "rl_agent_config.json")))
cfg["max_len"] = 512; cfg["head_max_len"] = 192

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from sql_cases import QUESTIONS
EFF = ["cheap", "moderate", "heavy"]

def to_internal(q):
    crit = q.get("criteria"); return {"t": q["type"], "ins": q["instructions"], "crit": {c: None for c in crit} if isinstance(crit, list) and q["type"] == "choice" else crit}

def items_of(ex):
    state = {"schema": ex["schema"], "intent": ex["intent"], "sql": ex["sql"]}
    gold = {"safe": [0.0, 1.0] if ex["safe"] else [1.0, 0.0], "correct": [0.0, 1.0] if ex["correct"] else [1.0, 0.0],
            "cost": [1.0 if i == ex["cost"] else 0.0 for i in range(3)], "kind": [1.0 if k == ex["kind"] else 0.0 for k in QUESTIONS["kind"]["criteria"]]}
    out = []
    for qid, q in QUESTIONS.items():
        qi = to_internal(q); seq, markers = build_sequence(tok, state, qi, cfg["max_len"], cfg["head_max_len"])
        tgt = gold[qid]; sm = 0.9 * torch.tensor(tgt) + 0.1 / len(tgt)   # label smoothing
        out.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["type"]], "target": sm.tolist(), "label": int(torch.tensor(tgt).argmax())})
    return out

def collate(items, pad):
    n, L = len(items), max(len(i["ids"]) for i in items); kmax = max(len(i["markers"]) for i in items)
    ids = torch.full((n, L), pad, dtype=torch.long); att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long); mmask = torch.zeros((n, kmax), dtype=torch.bool); tgt = torch.zeros((n, kmax))
    for i, it in enumerate(items):
        ids[i, :len(it["ids"])] = torch.tensor(it["ids"]); att[i, :len(it["ids"])] = 1
        k = len(it["markers"]); mpos[i, :k] = torch.tensor(it["markers"]); mmask[i, :k] = True; tgt[i, :k] = torch.tensor(it["target"])
    return dict(input_ids=ids, attention_mask=att, marker_pos=mpos, marker_mask=mmask, target=tgt, qtype=torch.tensor([i["qtype"] for i in items]))

examples = json.load(open("train_examples.json"))
items = [it for ex in examples for it in items_of(ex)]
if MAX_ITEMS: items = items[:MAX_ITEMS]
print(f"device={dev} items={len(items)} epochs={EPOCHS} micro={MICRO} accum={ACCUM}", flush=True)

model = build_model(cfg, encoder_dir=os.path.join(mdir, "encoder"))
model.load_state_dict(load_file(os.path.join(mdir, "model.safetensors")), strict=True)
model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.head_checkpointing = True
model.to(dev).train()
enc = [p for n, p in model.named_parameters() if "encoder." in n]; head = [p for n, p in model.named_parameters() if "encoder." not in n]
opt = torch.optim.AdamW([{"params": enc, "lr": 2.5e-5}, {"params": head, "lr": 1e-4}], weight_decay=0.01)
total = max(1, (len(items) // (MICRO * ACCUM)) * EPOCHS); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total, eta_min=1e-6)
G, S0, S1 = 4, 0.4, 0.1
t0 = time.time()
for ep in range(EPOCHS):
    random.seed(42 + ep); random.shuffle(items); sigma = S0 + (S1 - S0) * ep / max(1, EPOCHS - 1); tot = nb = 0; opt.zero_grad(set_to_none=True)
    for b in range(0, len(items), MICRO):
        batch = collate(items[b:b + MICRO], tok.pad_token_id)
        logits, act = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev), batch["marker_pos"].to(dev), batch["marker_mask"].to(dev), batch["qtype"].to(dev))
        logits = logits.float(); mask = batch["marker_mask"].to(dev); k = mask.sum(-1, keepdim=True).float(); target = batch["target"].to(dev)
        eps = torch.randn((G,) + logits.shape, device=dev) * sigma * mask; eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
        z = logits.detach().unsqueeze(0) + eps; q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
        with torch.no_grad():
            r = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(dev), mask, w_sph=0.75, w_rps=1.0); adv = r - r.mean(0, keepdim=True); adv = adv / (adv.std() + 1e-6)
        logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
        loss_rl = -(adv * logp).mean(); loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
        loss = (loss_rl + loss_ce) / ACCUM + 0.0 * act.sum(); loss.backward(); nb += 1; tot += loss.item() * ACCUM
        if nb % ACCUM == 0 or b + MICRO >= len(items):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if nb % 25 == 0: print(f"ep{ep+1} step{nb}/{len(items)//MICRO} loss={tot/nb:.4f} reward={r.mean().item():.3f} {time.time()-t0:.0f}s", flush=True)
        if STEP_TIMING and nb >= int(STEP_TIMING): print(f"{(time.time()-t0)/nb:.2f}s/step"); sys.exit()
    print(f"=== epoch {ep+1} avg loss {tot/max(1,nb):.4f} {time.time()-t0:.0f}s", flush=True)
    os.makedirs(OUT, exist_ok=True)
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, os.path.join(OUT, "model.safetensors"))
    model.encoder.config.save_pretrained(os.path.join(OUT, "encoder")); tok.save_pretrained(os.path.join(OUT, "tokenizer"))
    cfg2 = dict(cfg, fine_tuned=True, model_name="laya-sql-review", temperature=[1.0, 1.0, 1.0], temperature_by_options={})
    json.dump({"epoch": ep+1}, open(os.path.join(OUT, "meta.json"), "w"))
    json.dump(cfg2, open(os.path.join(OUT, "rl_agent_config.json"), "w"), indent=2)
print("saved", OUT)
