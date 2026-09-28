"""Async difficulty HARDENER for the synthetic queries (gold-preserving).

Standard label-invariant difficulty recipe (PromptBench-style noise + tool-use
distractors + persona/context framing), applied while keeping the programmatic
gold valid. For each query it asks a strong model (via the LLM_* environment variables) to
rewrite the request so it:
  1. opens with a PERSONA + situational CONTEXT sentence irrelevant to the actions;
  2. still contains ALL required actions with EVERY concrete value verbatim;
  3. adds 1-2 DISTRACTORS — remarks that sound like asks but map to NO tool
     (chit-chat / hypotheticals / out-of-scope), so a correct agent ignores them;
  4. uses filler and non-linear ordering.

Pair with the eval's SET-EQUALITY mode (`SYN_SETEQ=1`): the agent gets the full
noisy message once (no teacher-forcing) and must emit EXACTLY the gold call
multiset — so spurious distractor-induced calls are penalized.

Guard: every concrete gold value must remain recoverable (grounding >= 0.9) else
the row keeps its un-hardened phrasing.

Run:  LLM_MODEL=gemini-3.5-flash HARD_SAMPLE=30 python -m benchmarks.synthetic.generation.harden
Out:  benchmarks/synthetic/data/generated/queries_hard.jsonl (released as data/queries.jsonl)
"""
import asyncio
import json
import os
import random
import sys

from openai import AsyncOpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .llm_config import llm_client_kwargs
from .gen_complex import gold_values, grounded_fraction, spec
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")   # benchmarks/synthetic/data

IN = os.environ.get("HARD_IN", os.path.join(DATA, "generated", "queries_natural.jsonl"))
if not os.path.exists(IN):
    IN = os.path.join(DATA, "generated", "queries_templated.jsonl")
OUT = os.environ.get("HARD_OUT", os.path.join(DATA, "generated", "queries_hard.jsonl"))  # released as data/queries.jsonl
BASE_URL, API_KEY, MODEL = llm_client_kwargs()
SAMPLE = int(os.environ.get("HARD_SAMPLE", "30"))
WORKERS = int(os.environ.get("HARD_WORKERS", "10"))
TYPES = os.environ.get("HARD_TYPES", "single,compound,compositional").split(",")
MAXTOK = int(os.environ.get("HARD_MAXTOK", "1500"))
SEED = int(os.environ.get("HARD_SEED", "11"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

SYS = (
    "You create a HARDER benchmark version of a user request to stress-test a "
    "tool-calling assistant. You are given the EXACT tool actions the assistant "
    "must perform. Write ONE realistic but NOISY user message that still requires "
    "EXACTLY those actions and nothing more.\n"
    "Weave in, in a natural casual flow:\n"
    "1) PERSONA + CONTEXT: open with a sentence about who the user is / their mood "
    "/ the weather / what happened today — IRRELEVANT to the actions.\n"
    "2) ALL required actions, each with EVERY concrete value verbatim (numbers as "
    "digits; names, codes, titles, times exactly).\n"
    "3) ONE or TWO DISTRACTORS: remarks that sound like requests but must NOT be "
    "acted on and do NOT correspond to any device/tool action — e.g. 'remind me to "
    "call mom later', 'I wonder if it'll rain', 'I should really clean the garage "
    "someday'. Never phrase a distractor as another doable action with concrete "
    "settings.\n"
    "4) Filler words, hedging, and a non-linear order.\n"
    "HARD RULES: keep every required value exactly; do NOT drop, merge, add, or "
    "alter any required action; distractors must map to no real action. Output "
    "ONLY the user message (2-5 sentences). No quotes, no lists, no explanation."
)


async def harden(sem, row):
    async with sem:
        user = (f"Required tool actions (perform exactly these):\n{spec(row)}\n\n"
                f"Write the noisy user message:")
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.9, max_tokens=MAXTOK,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip().strip('"').strip()
        except Exception as e:
            text = ""
            row["_err"] = str(e)[:200]
    vals = gold_values(row)
    gf = grounded_fraction(text, vals) if text else 0.0
    row.setdefault("query_base", row["query"])
    row["query_easy"] = row.get("query")          # the natural (un-hardened) phrasing
    row["query_hard"] = text
    row["grounded_frac"] = round(gf, 3)
    row["query"] = text if (text and gf >= 0.9) else row.get("query_base")
    row["gen_model"] = MODEL
    return row


async def main():
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    pool = [r for r in rows if r["type"] in TYPES]
    rng.shuffle(pool)
    pool = pool[:SAMPLE]
    sem = asyncio.Semaphore(WORKERS)
    done = await asyncio.gather(*(harden(sem, r) for r in pool))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in done if r["grounded_frac"] >= 0.9)
    avg = sum(r["grounded_frac"] for r in done) / max(1, len(done))
    print(f"model={MODEL}  hardened={sum(1 for r in done if r['query_hard'])}/{len(done)}  "
          f"grounded>=0.9: {ok}/{len(done)}  avg={avg:.3f}  src={os.path.basename(IN)}  -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
