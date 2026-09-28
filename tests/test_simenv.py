"""SimEnv gate: a perfect model driven through the SIMULATED ENVIRONMENT
must score exact=1.0 under every operator spec, and the error paths must behave
(reject + allow recovery) under hard control.

Unlike tests/test_ops (pure decode round-trip), this exercises SimEnv semantics:
txn ids returned by begin, resolver lookups (exact + fuzzy), commit rollback on
missing required args, hard-control rejections.

Run:  python -m tests.test_simenv
"""
import json
import os
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from eval import common as ev
from toolschema.operators import compile_spec
from benchmarks.synthetic.opspecs import DOMAINS, specs_for
from tests.test_ops import by_group, queries
from toolschema.simenv import SimEnv


def key(n, a):
    return (n, tuple(sorted(ev.clean(a).items())))


def perfect_episode(comp, golds, control="hard"):
    """Drive SimEnv with the oracle sequence; return (exact, rejected, flags)."""
    env = SimEnv(comp.call_map, control=control)
    for name, args in comp.encode_episode(golds):
        env.call(name, args)
    flags = env.finish()
    gset = Counter(key(g["name"], g["arguments"]) for g in golds)
    aset = Counter(key(n, a) for n, a in env.native_actions)
    return int(gset == aset), env.n_rejected, flags


def unit_error_paths():
    tools = by_group["smart_home"]
    d = next(dd for dd in DOMAINS if dd["name"] == "smart_home")
    spec = {"ops": [{"op": "curry", "tool": "set_fan"},
                    {"op": "split_predicate", "tool": "set_brightness",
                     "param": "brightness", "cuts": [50]},
                    {"op": "merge", "tools": ["set_blinds", "set_outlet"]}]}
    comp = compile_spec(tools, spec)
    env = SimEnv(comp.call_map, control="hard")

    r = json.loads(env.call("no_such_fn", {}))
    assert "error" in r, r
    r = json.loads(env.call("execute", {"operation": "set_room_light"}))
    assert "error" in r, ("smuggle should be rejected", r)
    pred = [t["name"] for t in comp.tools if t["name"].startswith("set_brightness__")]
    r = json.loads(env.call(pred[0], {"room": "kitchen", "brightness": 999999}))
    assert "error" in r, ("pred violation should be rejected", r)
    r = json.loads(env.call("set_set_fan__speed", {"txn_id": "txn_9", "speed": "low"}))
    assert "error" in r, ("set on unknown txn should fail", r)
    r = json.loads(env.call("begin_set_fan", {}))
    tid = r["txn_id"]
    r = json.loads(env.call("commit_set_fan", {"txn_id": tid}))
    assert "error" in r and "missing required" in r["error"], ("early commit", r)
    for p, v in [("room", "office"), ("speed", "medium"), ("direction", "forward")]:
        assert json.loads(env.call(f"set_set_fan__{p}", {"txn_id": tid, p: v})).get("status") == "set"
    r = json.loads(env.call("commit_set_fan", {"txn_id": tid}))
    assert r.get("status") == "committed", ("recovered commit must succeed", r)
    assert env.native_actions == [("set_fan", {"room": "office", "speed": "medium",
                                               "direction": "forward"})], env.native_actions
    assert not [f for f in env.finish() if f[0] == "dangling_txn"], "no dangling txn expected"
    print("error-path unit: OK (reject + recovery + rollback)")

    # fuzzy resolver
    d2 = next(dd for dd in DOMAINS if any("_pool" in pd for t in dd["tools"]
                                          for pd in t["params"].values()))
    for t in d2["tools"]:
        pools = [(p, pd["_pool"]) for p, pd in t["params"].items() if "_pool" in pd]
        if pools:
            p, pool = pools[0]
            comp2 = compile_spec(by_group[d2["name"]],
                                 {"ops": [{"op": "indirect", "tool": t["name"],
                                           "param": p, "values": pool}]})
            env2 = SimEnv(comp2.call_map, control="hard")
            rname = f"resolve_{t['name']}__{p}"
            v = pool[0]
            ok = json.loads(env2.call(rname, {"query": str(v).upper()}))
            assert "handle" in ok, ("case-insensitive resolve failed", v, ok)
            miss = json.loads(env2.call(rname, {"query": "zzz_no_such_thing_qq"}))
            assert "error" in miss, miss
            print(f"resolver unit: OK ({d2['name']}.{t['name']}.{p})")
            return


def main():
    unit_error_paths()
    checked, fails = 0, []
    for d in DOMAINS:
        g = d["name"]
        for tag, spec in specs_for(d, queries[g]).items():
            if not spec["ops"]:
                continue
            comp = compile_spec(by_group[g], spec)
            for q in queries[g][:40]:              # SimEnv gate: subset per domain
                exact, rej, flags = perfect_episode(comp, q["gold_calls"])
                checked += 1
                if not exact or rej or flags:
                    fails.append((g, tag, q["id"], exact, rej, flags))
    print(f"perfect-model episodes through SimEnv: {checked}")
    if fails:
        print(f"FAILURES: {len(fails)}")
        for f in fails[:10]:
            print("  ", f)
        sys.exit(1)
    print("OK: SimEnv executes every spec's oracle sequence to exact=1.0")


if __name__ == "__main__":
    main()
