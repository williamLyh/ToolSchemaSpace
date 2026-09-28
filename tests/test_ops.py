"""Oracle round-trip gate for operator specs (the full operator space, including
call SEQUENCES).

For every domain × spec × gold: encode the golds into the variant's canonical
call sequence (Compiled.encode_episode) and fold it back through SeqDecoder.
The decoded native calls must equal the golds and the decoder must raise zero
flags — a perfect model scores 1.0 under every spec, so any measured accuracy
difference is a real schema-usage effect (verified, never assumed).

Run:  python -m tests.test_ops
"""
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev            # clean(): normalized arg compare
from toolschema.operators import compile_spec
from benchmarks.synthetic.opspecs import DOMAINS, specs_for

TOOLS = json.load(open(os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json"), encoding="utf-8"))
QPATH = os.environ.get("SYN_QUERIES",
                       os.path.join(ROOT, "benchmarks", "synthetic", "data", "queries.jsonl"))

by_group = defaultdict(list)
for t in TOOLS:
    by_group[t["group"]].append({"name": t["name"], "description": t["description"],
                                 "parameters": t["parameters"]})
queries = defaultdict(list)
for l in open(QPATH, encoding="utf-8"):
    r = json.loads(l)
    queries[r["domain"]].append(r)


def roundtrip(comp, golds):
    seq = comp.encode_episode(golds)
    native, flags = comp.decode_seq(seq)
    if flags:
        return False, f"flags={flags}"
    if len(native) != len(golds):
        return False, f"{len(native)} native calls vs {len(golds)} golds"
    for (nm, na), g in zip(native, golds):
        if nm != g["name"] or ev.clean(na) != ev.clean(g["arguments"]):
            return False, f"{nm}({na}) != {g['name']}({g['arguments']})"
    return True, ""


def test_arg_lift_unit():
    """arg_lift has no natural substrate in synthetic (all-flat schemas): unit-test
    it on a hand-made nested tool with nested golds."""
    nested = [{"name": "book", "description": "Book a thing.",
               "parameters": {"type": "object", "properties": {
                   "city": {"type": "string", "description": "City."},
                   "options": {"type": "object", "description": "Extras.",
                               "properties": {"seat": {"type": "string", "description": "Seat."},
                                              "bags": {"type": "integer", "description": "Bags."}},
                               "required": ["seat"]}},
                   "required": ["city", "options"]}}]
    comp = compile_spec(nested, {"ops": [{"op": "arg_lift", "tool": "book", "param": "options"}]})
    props = comp.tools[-1]["parameters"]["properties"]
    assert set(props) == {"city", "seat", "bags"}, props
    gold = [{"name": "book", "arguments": {"city": "Paris",
                                           "options": {"seat": "12A", "bags": 2}}}]
    seq = comp.encode_episode(gold)
    assert seq == [("book", {"city": "Paris", "seat": "12A", "bags": 2})], seq
    native, flags = comp.decode_seq(seq)
    assert not flags and native[0][0] == "book", (native, flags)
    assert native[0][1] == {"city": "Paris", "options": {"seat": "12A", "bags": 2}}, native
    print("arg_lift unit: OK")


def main():
    test_arg_lift_unit()
    checked = Counter()
    fails = []
    n_tools = Counter()
    for d in DOMAINS:
        g = d["name"]
        tools = by_group[g]
        assert tools, f"no tools for domain {g}"
        for tag, spec in specs_for(d, queries[g]).items():
            if not spec["ops"]:
                continue
            comp = compile_spec(tools, spec)
            n_tools[tag] += len(comp.tools)
            for q in queries[g]:
                ok, why = roundtrip(comp, q["gold_calls"])
                checked[tag] += 1
                if not ok:
                    fails.append((g, tag, q["id"], why))
    # global-spec gate: the merged multi-group spec files must
    # behave identically under group filtering — tool names collide across
    # domains (e.g. set_volume), which is exactly what this catches.
    merged = {}
    for d in DOMAINS:
        for tag, spec in specs_for(d, queries[d["name"]]).items():
            for o in spec["ops"]:
                o["group"] = d["name"]
            merged.setdefault(tag, []).extend(spec["ops"])
    for tag, ops in merged.items():
        for d in DOMAINS:
            g = d["name"]
            comp = compile_spec(by_group[g], {"ops": ops}, group=g)
            for q in queries[g][:25]:
                ok, why = roundtrip(comp, q["gold_calls"])
                checked[f"global_{tag}"] += 1
                if not ok:
                    fails.append((g, f"global_{tag}", q["id"], why))

    print(f"checked (gold-episode x spec): {dict(checked)}  total={sum(checked.values())}")
    print(f"variant catalog sizes (sum over domains): {dict(n_tools)}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:15]:
            print("  ", f)
        sys.exit(1)
    print("OK: operator-spec scoring is variant-invariant (all specs round-trip)")


if __name__ == "__main__":
    main()
