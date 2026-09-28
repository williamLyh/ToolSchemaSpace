"""Sample tool combinations, then synthesize (hardened) queries for them.

Motivation: the schema is now big (168 fns, larger enums). The hand-authored
routines only exercise a fixed slice of tool combinations; to cover the schema
broadly we SAMPLE which tools a query uses (a random within-domain K-subset, K
weighted toward multi-tool), sample valid args for each (respecting enums / NUM
pools / free-string pools), and build the programmatic gold. Then a strong model
(set with LLM_BASE_URL / LLM_API_KEY / LLM_MODEL) writes a realistic HARDENED user request for that
exact gold (persona + context + out-of-scope distractors + noise), so the gold
stays valid and the style matches the canonical `queries_hard`.

Grounding guard: every concrete gold value must remain recoverable (>=0.9) else
the row keeps a plain deterministic listing as its query.

Run:  LLM_MODEL=gemini-3.5-flash GEN_SAMPLE=1500 GEN_WORKERS=12 python -m benchmarks.synthetic.generation.gen_sampled
Out:  benchmarks/synthetic/data/generated/queries_sampled.jsonl
"""
import asyncio
import json
import os
import random
import sys

from openai import AsyncOpenAI

from .llm_config import llm_client_kwargs
from ..domains import DOMAINS
from ..build import candidate_values
from .gen_complex import gold_values, grounded_fraction, spec

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUT = os.path.join(DATA, "generated", "queries_sampled.jsonl")
BASE_URL, API_KEY, MODEL = llm_client_kwargs()
SAMPLE = int(os.environ.get("GEN_SAMPLE", "30"))       # total queries across all domains
WORKERS = int(os.environ.get("GEN_WORKERS", "10"))
MAXTOK = int(os.environ.get("GEN_MAXTOK", "1500"))
SEED = int(os.environ.get("GEN_SEED", "13"))
# K = number of tools a query uses; weighted toward small multi-tool combos
K_CHOICES = [1, 2, 3, 4, 5]
K_WEIGHTS = [0.12, 0.28, 0.30, 0.18, 0.12]

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

SYS = (
    "You create a realistic user request to benchmark a tool-calling assistant. "
    "You are given the EXACT tool actions the assistant must perform. Write ONE "
    "natural but NOISY user message that requires EXACTLY those actions and nothing "
    "more.\n"
    "Weave in, in a casual flow:\n"
    "1) a short PERSONA + situational CONTEXT sentence IRRELEVANT to the actions;\n"
    "2) ALL required actions, each with EVERY concrete value verbatim (numbers as "
    "digits; names, colors, codes, titles, times exactly — e.g. 'crimson', not 'red');\n"
    "3) ONE or TWO DISTRACTORS: remarks that sound like asks but map to NO tool "
    "(chit-chat, hypotheticals, out-of-scope wishes); never phrase a distractor as "
    "another doable action with settings;\n"
    "4) filler and non-linear order.\n"
    "HARD RULES: keep every required value exactly; do not drop/merge/add/alter any "
    "action; distractors must map to no real action. Output ONLY the user message "
    "(2-5 sentences). No quotes, no lists, no explanation."
)


def sample_combo(rng, d):
    """Pick K distinct tools in domain d and sample valid args for each -> gold."""
    tools = d["tools"]
    k = min(rng.choices(K_CHOICES, weights=K_WEIGHTS)[0], len(tools))
    chosen = rng.sample(tools, k)
    calls = []
    for t in chosen:
        args = {p: rng.choice(candidate_values(spec_)) for p, spec_ in t["params"].items()}
        calls.append({"name": t["name"], "arguments": args})
    return calls


def naive_base(calls):
    parts = []
    for c in calls:
        a = " ".join(f"{k} {v}" for k, v in c["arguments"].items())
        parts.append(f"{c['name'].replace('_', ' ')} {a}")
    return "Please do the following: " + "; ".join(parts) + "."


async def synth(sem, row):
    async with sem:
        user = f"Required tool actions (do exactly these):\n{spec(row)}\n\nWrite the user message:"
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.9, max_tokens=MAXTOK,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip().strip('"').strip()
        except Exception as e:
            text = ""
            row["_err"] = str(e)[:200]
    gf = grounded_fraction(text, gold_values(row)) if text else 0.0
    row["query_hard"] = text
    row["grounded_frac"] = round(gf, 3)
    row["query"] = text if (text and gf >= 0.9) else row["query_base"]
    row["gen_model"] = MODEL
    return row


async def main():
    rng = random.Random(SEED)
    per = max(1, SAMPLE // len(DOMAINS))
    rows = []
    for d in DOMAINS:
        seen, made, tries = set(), 0, 0
        while made < per and tries < per * 10:
            tries += 1
            calls = sample_combo(rng, d)
            key = tuple(sorted((c["name"], tuple(sorted((k, str(v)) for k, v in c["arguments"].items()))) for c in calls))
            if key in seen:
                continue
            seen.add(key)
            row = {"id": f"{d['name']}__sampled__{made}", "domain": d["name"], "system": d["system"],
                   "type": "single" if len(calls) == 1 else "compound",
                   "gold_calls": calls, "n_calls": len(calls)}
            row["query_base"] = naive_base(calls)
            rows.append(row)
            made += 1
    sem = asyncio.Semaphore(WORKERS)
    done = await asyncio.gather(*(synth(sem, r) for r in rows))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in done if r["grounded_frac"] >= 0.9)
    from collections import Counter
    kc = Counter(r["n_calls"] for r in done)
    print(f"model={MODEL}  made={len(done)}  grounded>=0.9: {ok}/{len(done)}  "
          f"n_calls={dict(sorted(kc.items()))}  -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
