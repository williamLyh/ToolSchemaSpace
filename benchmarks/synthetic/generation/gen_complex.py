"""Async, gold-conditioned COMPLEX query generation with an online LLM.

Instead of the template + light-paraphrase pipeline, this asks a strong model
(default `gemini-3.5-flash`, via any OpenAI-compatible endpoint set with the
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL environment variables) to write a richer, more natural, multi-step user request for a
*fixed* set of gold tool calls. The gold is conditioned on, never invented, so the
programmatic gold stays valid and the queries remain gradable.

Quality guard: every concrete gold value (number / enum / free string) must still
be recoverable from the generated text (lenient grounding check). Rows that fail
keep their original phrasing; we record the grounded fraction either way.

Run (after setting the LLM_* environment variables):
  LLM_MODEL=gemini-3.5-flash python -m benchmarks.synthetic.generation.gen_complex            # 30-row trial
  GEN_SAMPLE=400 GEN_WORKERS=12 python -m benchmarks.synthetic.generation.gen_complex     # scale up
Output: benchmarks/synthetic/data/generated/queries_natural.jsonl  (id, type, gold_calls, query_base,
        query_llm, query, grounded_frac, gen_model).
"""
import asyncio
import json
import os
import random
import re
import sys

from openai import AsyncOpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from .llm_config import llm_client_kwargs
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")   # benchmarks/synthetic/data

IN = os.path.join(DATA, "generated", "queries_templated.jsonl")
OUT = os.path.join(DATA, "generated", "queries_natural.jsonl")
BASE_URL, API_KEY, MODEL = llm_client_kwargs()
SAMPLE = int(os.environ.get("GEN_SAMPLE", "30"))
WORKERS = int(os.environ.get("GEN_WORKERS", "8"))
TYPES = os.environ.get("GEN_TYPES", "compound,compositional").split(",")
MAXTOK = int(os.environ.get("GEN_MAXTOK", "1400"))     # reasoning models burn ~300-500 tok before output
SEED = int(os.environ.get("GEN_SEED", "7"))

client = AsyncOpenAI(base_url=BASE_URL, api_key=API_KEY)

SYS = (
    "You write realistic, natural user requests for a voice/agent assistant. You "
    "are given an ORDERED list of tool calls the assistant must perform. Write ONE "
    "user message (1-3 sentences, conversational, with a little real-life framing) "
    "that asks for EXACTLY those actions, in that order.\n"
    "HARD RULES:\n"
    "- Include EVERY concrete detail exactly: each device/target, room/zone/seat, "
    "numeric value, on/off, mode, color, name, code, time, etc. Do NOT add, drop, "
    "merge, or change any action or value.\n"
    "- Keep numbers as digits. Keep names/codes/titles verbatim.\n"
    "- Sound human (you may add brief context like 'I'm heading out' or 'it's "
    "freezing'), but do not invent extra actions.\n"
    "- Output ONLY the user message: no quotes, no lists, no explanation."
)


def spec(row):
    lines = []
    for i, c in enumerate(row["gold_calls"], 1):
        a = ", ".join(f"{k}={v}" for k, v in c["arguments"].items())
        lines.append(f"{i}. {c['name']}({a})")
    return "\n".join(lines)


def gold_values(row):
    """Concrete values that should be recoverable from the text (skip booleans)."""
    vals = []
    for c in row["gold_calls"]:
        for v in c["arguments"].values():
            if isinstance(v, bool):
                continue
            vals.append(v)
    return vals


def _norm(s):
    return re.sub(r"[\s_]+", " ", str(s).lower()).strip()


def grounded_fraction(text, vals):
    if not vals:
        return 1.0
    t = _norm(text)
    hit = 0
    for v in vals:
        nv = _norm(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            # number must appear as a token
            if re.search(rf"(?<!\d){re.escape(str(v))}(?!\d)", t):
                hit += 1
        elif nv and nv in t:
            hit += 1
    return hit / len(vals)


async def generate(sem, row):
    async with sem:
        user = (f"Tool calls to request (in order):\n{spec(row)}\n\n"
                f"Write the user message:")
        try:
            r = await client.chat.completions.create(
                model=MODEL, temperature=0.8, max_tokens=MAXTOK,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            text = (r.choices[0].message.content or "").strip().strip('"').strip()
        except Exception as e:
            text = ""
            row["_err"] = str(e)[:200]
    vals = gold_values(row)
    gf = grounded_fraction(text, vals) if text else 0.0
    row["query_llm"] = text
    row["grounded_frac"] = round(gf, 3)
    # adopt the LLM query only if it preserves the concrete values well
    row["query"] = text if (text and gf >= 0.9) else row.get("query_base") or row["query"]
    row["gen_model"] = MODEL
    return row


async def main():
    rng = random.Random(SEED)
    rows = [json.loads(l) for l in open(IN, encoding="utf-8") if l.strip()]
    pool = [r for r in rows if r["type"] in TYPES]
    rng.shuffle(pool)
    pool = pool[:SAMPLE]
    for r in pool:
        r.setdefault("query_base", r["query"])
    sem = asyncio.Semaphore(WORKERS)
    done = await asyncio.gather(*(generate(sem, r) for r in pool))
    with open(OUT, "w", encoding="utf-8") as f:
        for r in done:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in done if r["grounded_frac"] >= 0.9)
    avg = sum(r["grounded_frac"] for r in done) / max(1, len(done))
    nonempty = sum(1 for r in done if r["query_llm"])
    print(f"model={MODEL}  generated={nonempty}/{len(done)}  "
          f"grounded>=0.9: {ok}/{len(done)}  avg_grounded={avg:.3f}  -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
