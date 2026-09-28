"""Operator-spec compiler — the operator space beyond the 1-D merge/split ladder.

A *variant spec* is a JSON dict {"ops": [...]} listing operator applications.
`compile_spec(tools, spec)` turns a canonical catalog (list of
{name, description, parameters{type,properties,required}}) into a `Compiled`:

    .tools              forward catalog (canonical shape; adapter converts to OpenAI)
    .call_map           per-variant-function decode entries (superset of the legacy
                        schema_transform format: extra entries carry a "kind" tag)
    .encode_episode(gold_calls) -> [(vname, vargs), ...]
                        ORACLE: canonical call sequence expressing the golds in
                        this variant's vocabulary (the equivalence witness)
    .decoder() -> SeqDecoder    stateful sequence decoder: feed model calls one
                        at a time; meta calls (resolve/begin/set/list) yield no
                        native action, commits emit one; flush() reports flags
    .decode_seq(calls) -> (native_calls, flags)   convenience fold

Operators (spec entries; `tool`/`tools` refer to native operation names; ops
naming tools absent from the catalog are ignored, so ONE spec can be applied
across all groups):

    {"op":"merge", "tools":[...] , "into":"execute", "encoding":"flat|nested|union"}
                                                         # flat (default) = operation + namespaced args
                                                         # nested = operation + arguments={native args}
                                                         # union  = one optional object key per member
    {"op":"split_enum", "tool":t, "params":[p,...]}      # bake enum/bool → names
    {"op":"split_predicate", "tool":t, "param":p, "cuts":[c1,...], "labels":[...]}
                                                         # interval partition → names
    {"op":"arg_lower", "tool":t, "into":"options", "params":[p,...]}  # nest
    {"op":"arg_lift", "tool":t, "param":p}               # flatten nested object
    {"op":"indirect", "tool":t, "param":p, "values":[...]}
                                                         # free value → resolver+handle
    {"op":"curry", "tool":t}                             # begin/set*/commit

Composition: ops are normalized into a per-tool plan, then compiled in one
pass. Restrictions (asserted): a merged tool takes no other operator; a curried
tool cannot also bake/pred-split (name×time explosion is out of scope).
Equivalence is NEVER assumed: every spec must pass the oracle round-trip
(tests/test_ops.py) before it is used in an eval.
"""

import hashlib
import itertools
import json
import random as _random

from .merge_encodings import (ENCODINGS, decode_merged, encode_merged,
                             execute_tool_nested, execute_tool_union,
                             member_param_index, validate_merged)
from .schema_transform import (_execute_tool, _gprops, _greq,
                              _is_enumerable, _named_tools_mapped, _san)

TXN_PREFIX = "txn_"


def handle_of(tool, param, value):
    """Deterministic opaque handle for an indirected value (oracle + SimEnv share it)."""
    h = hashlib.sha1(f"{tool}::{param}::{value}".encode()).hexdigest()[:8]
    return f"h_{h}"


# --------------------------------------------------------------------------- #
# spec → per-tool plan
# --------------------------------------------------------------------------- #
def _norm_plan(tools, spec, group=None):
    byname = {t["name"]: t for t in tools}
    merges = {}                                    # into_name -> {"members": [...], "encoding": ...}
    plan = {t["name"]: {"baked": [], "pred": {}, "pack": [], "lift": [],
                        "indirect": {}, "curry": False,
                        "rename": None, "nodesc": False, "shuffle": None} for t in tools}
    for o in spec.get("ops", []):
        # tool names are NOT unique across groups (e.g. set_volume): ops stamped
        # with a "group" apply only to that group's catalog. Ops without a stamp
        # apply wherever the tool name matches (single-group specs).
        if o.get("group") is not None and group is not None and o["group"] != group:
            continue
        kind = o["op"]
        if kind == "merge":
            members = [n for n in o["tools"] if n in byname]
            encoding = o.get("encoding", "flat")
            if encoding not in ENCODINGS:
                raise ValueError(f"unknown merge encoding: {encoding}")
            if members:
                into = o.get("into", "execute")
                slot = merges.setdefault(into, {"members": [], "encoding": encoding})
                if slot["encoding"] != encoding:
                    raise ValueError(
                        f"conflicting encodings for {into}: "
                        f"{slot['encoding']} vs {encoding}")
                slot["members"].extend(members)
            continue
        t = o.get("tool")
        if t not in byname:
            continue
        p = plan[t]
        if kind == "split_enum":
            for prm in o["params"]:
                pd = _gprops(byname[t]).get(prm)
                assert pd is not None and _is_enumerable(pd), f"split_enum: {t}.{prm} not enumerable"
                if prm not in p["baked"]:
                    p["baked"].append(prm)
        elif kind == "split_predicate":
            cuts = sorted(o["cuts"])
            assert cuts, f"split_predicate: {t}.{o['param']} needs cuts"
            pd = _gprops(byname[t]).get(o["param"])
            assert pd is not None and pd.get("type") in ("integer", "number"), \
                f"split_predicate: {t}.{o['param']} missing or not numeric"
            p["pred"][o["param"]] = {"cuts": cuts, "labels": o.get("labels")}
        elif kind == "arg_lower":
            p["pack"].append({"into": o.get("into", "options"), "params": list(o["params"])})
        elif kind == "arg_lift":
            p["lift"].append(o["param"])
        elif kind == "indirect":
            p["indirect"][o["param"]] = {"values": list(o.get("values", []))}
        elif kind == "curry":
            p["curry"] = True
        elif kind == "rename":                     # surface: opaque/other fn name
            p["rename"] = o["to"]
        elif kind == "strip_desc":                 # surface: drop all descriptions
            p["nodesc"] = True
        elif kind == "shuffle_params":             # surface: permute param order
            p["shuffle"] = int(o.get("seed", 0))
        else:
            raise ValueError(f"unknown operator: {kind}")
    merged_set = {n for slot in merges.values() for n in slot["members"]}
    for n in merged_set:
        q = plan[n]
        assert not (q["baked"] or q["pred"] or q["pack"] or q["lift"]
                    or q["indirect"] or q["curry"] or q["rename"]
                    or q["nodesc"] or q["shuffle"] is not None), \
            f"merged tool {n} takes no other op"
    for n, q in plan.items():
        assert not (q["curry"] and (q["baked"] or q["pred"])), \
            f"curry({n}) cannot combine with split ops"
        assert not (q["pack"] and q["lift"]), \
            f"arg_lower+arg_lift on the same tool ({n}) is unsupported"
    return plan, merges


# --------------------------------------------------------------------------- #
# per-operator schema builders
# --------------------------------------------------------------------------- #
def _pred_intervals(cuts, labels=None):
    """cuts [c1<..<ck] -> [(label, lo, hi)] with lo inclusive, hi exclusive."""
    edges = [None] + list(cuts) + [None]
    out = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if labels and i < len(labels):
            lab = labels[i]
        elif lo is None:
            lab = f"lt_{_san(hi)}"
        elif hi is None:
            lab = f"ge_{_san(lo)}"
        else:
            lab = f"{_san(lo)}_to_{_san(hi)}"
        out.append((lab, lo, hi))
    return out


def _in_interval(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return (lo is None or v >= lo) and (hi is None or v < hi)


def _apply_pack_lift(tool, packs, lifts):
    """arg_lower / arg_lift on one tool's schema; returns (tool', pack_map, lift_map)."""
    props = {k: dict(v) for k, v in _gprops(tool).items()}
    req = _greq(tool)
    pack_map = {}                                  # into -> [inner params]
    for pk in packs:
        inner = {p: props.pop(p) for p in pk["params"] if p in props}
        if not inner:
            continue
        inner_req = [p for p in pk["params"] if p in req]
        req = [r for r in req if r not in inner]
        props[pk["into"]] = {"type": "object", "properties": inner,
                             "required": inner_req,
                             "description": f"Grouped arguments: {', '.join(inner)}."}
        if inner_req:
            req.append(pk["into"])
        pack_map[pk["into"]] = list(inner)
    lift_map = {}                                  # lifted inner name -> source param
    for lp in lifts:
        pd = props.pop(lp, None)
        if not pd or pd.get("type") != "object":
            continue
        inner = pd.get("properties", {})
        for ip, ipd in inner.items():
            assert ip not in props, f"arg_lift collision: {tool['name']}.{ip}"
            props[ip] = dict(ipd)
            lift_map[ip] = lp
        req = [r for r in req if r != lp] + list(pd.get("required", []))
    t2 = {**tool, "parameters": {"type": "object", "properties": props, "required": req}}
    return t2, pack_map, lift_map


def _apply_indirect(tool, ind, resolvers, call_map):
    """Replace each indirected free param with an opaque-handle param + resolver tool."""
    props = {k: dict(v) for k, v in _gprops(tool).items()}
    req = _greq(tool)
    ind_map = {}                                   # ref param -> (orig param, resolver name)
    for prm, meta in ind.items():
        if prm not in props:
            continue
        rname = f"resolve_{tool['name']}__{prm}"
        table = {handle_of(tool["name"], prm, v): v for v in meta["values"]}
        resolvers.append({
            "name": rname,
            "description": (f"Look up the {prm} for `{tool['name']}` by name/description "
                            f"and return an opaque handle. Call this FIRST, then pass the "
                            f"returned handle to `{tool['name']}`."),
            "parameters": {"type": "object",
                           "properties": {"query": {"type": "string",
                                          "description": f"The {prm} as mentioned by the user."}},
                           "required": ["query"]},
        })
        call_map[rname] = {"kind": "resolver", "op": tool["name"], "param": prm,
                           "table": table, "namespaced": False, "fixed": {}}
        ref = f"{prm}_ref"
        props[ref] = {"type": "string",
                      "description": (f"Opaque handle for {prm}, returned by `{rname}`. "
                                      f"Do NOT pass the raw {prm} value.")}
        del props[prm]
        req = [ref if r == prm else r for r in req]
        ind_map[ref] = (prm, rname, table)
    t2 = {**tool, "parameters": {"type": "object", "properties": props, "required": req}}
    return t2, ind_map


def _strip_desc(tool):
    """Surface op: remove the function description and every param description."""
    def clean(props):
        out = {}
        for k, v in props.items():
            v = dict(v)
            v.pop("description", None)
            if v.get("properties"):
                v["properties"] = clean(v["properties"])
            out[k] = v
        return out
    pr = tool["parameters"]
    return {**tool, "description": "",
            "parameters": {**pr, "properties": clean(pr.get("properties", {}))}}


def _shuffle_params(tool, seed):
    """Surface op: deterministic permutation of the property (and required) order."""
    props = list(tool["parameters"].get("properties", {}).items())
    rng = _random.Random(seed * 1000003 + sum(map(ord, tool["name"])))
    rng.shuffle(props)
    req = set(tool["parameters"].get("required", []))
    return {**tool, "parameters": {**tool["parameters"], "properties": dict(props),
                                   "required": [p for p, _ in props if p in req]}}


def _curry_tools(tool, call_map):
    """begin_t() / set_t__p(txn_id, p) / commit_t(txn_id) triple for one tool."""
    t = tool["name"]
    txn = {"type": "string", "description": f"Transaction id returned by `begin_{t}`."}
    out = [{"name": f"begin_{t}",
            "description": (f"Start a `{t}` transaction. Returns a transaction id. "
                            f"Then set each argument with the set_ functions and finish "
                            f"with `commit_{t}`. {tool.get('description', '')}"),
            "parameters": {"type": "object", "properties": {}, "required": []}}]
    call_map[f"begin_{t}"] = {"kind": "curry_begin", "op": t, "namespaced": False, "fixed": {}}
    for prm, pd in _gprops(tool).items():
        out.append({"name": f"set_{t}__{prm}",
                    "description": f"Set `{prm}` on an open `{t}` transaction. {pd.get('description', '')}",
                    "parameters": {"type": "object",
                                   "properties": {"txn_id": dict(txn), prm: dict(pd)},
                                   "required": ["txn_id", prm]}})
        call_map[f"set_{t}__{prm}"] = {"kind": "curry_set", "op": t, "param": prm,
                                       "namespaced": False, "fixed": {}}
    out.append({"name": f"commit_{t}",
                "description": f"Commit the open `{t}` transaction. Nothing happens until this is called.",
                "parameters": {"type": "object", "properties": {"txn_id": dict(txn)},
                               "required": ["txn_id"]}})
    call_map[f"commit_{t}"] = {"kind": "curry_commit", "op": t, "required": _greq(tool),
                               "namespaced": False, "fixed": {}}
    return out


# --------------------------------------------------------------------------- #
# compiler
# --------------------------------------------------------------------------- #
class Compiled:
    def __init__(self, tools, call_map, enc_index):
        self.tools = tools
        self.call_map = call_map
        self._enc = enc_index                      # native op -> encoding recipe

    # ---- oracle ------------------------------------------------------------
    def encode_call(self, name, args, txn_no=1):
        """One gold call -> canonical variant call sequence (may be >1 step)."""
        r = self._enc[name]
        args = dict(args)
        pre = []
        if r["kind"] == "merged":
            return [encode_merged(r.get("encoding", "flat"), r["entry"], name, args)]
        # indirect: emit resolver calls first, swap value -> handle
        for ref, (prm, rname, _tab) in r["ind_map"].items():
            if prm in args:
                raw = args.pop(prm)
                pre.append((rname, {"query": str(raw)}))
                args[ref] = handle_of(name, prm, raw)
        base = r.get("base", name)
        if r["curry"]:
            tid = f"{TXN_PREFIX}{txn_no}"
            seq = pre + [(f"begin_{base}", {})]
            seq += [(f"set_{base}__{p}", {"txn_id": tid, p: v}) for p, v in args.items()]
            seq.append((f"commit_{base}", {"txn_id": tid}))
            return seq
        # name suffixes: baked enums then predicate intervals (spec order)
        fname, fixed = base, {}
        for p in r["baked"]:
            v = args.pop(p)
            fname += f"__{p}_{_san(v)}"
            fixed[p] = v
        for p, iv in r["pred"].items():
            seg = next((lab for lab, lo, hi in iv if _in_interval(args.get(p), lo, hi)), None)
            assert seg is not None, f"gold {name}.{p}={args.get(p)!r} outside predicate cover"
            fname += f"__{p}_{seg}"
        # packing: nest grouped args
        for into, inner in r["pack_map"].items():
            grp = {p: args.pop(p) for p in inner if p in args}
            if grp:
                args[into] = grp
        if r["lift_map"]:                          # lifted tool: gold args arrive nested
            flat = {}
            for k, v in args.items():
                if isinstance(v, dict) and any(src == k for src in r["lift_map"].values()):
                    flat.update(v)
                else:
                    flat[k] = v
            args = flat
        return pre + [(fname, args)]

    def encode_episode(self, gold_calls):
        seq, txn = [], 1
        for g in gold_calls:
            seq += self.encode_call(g["name"], g.get("arguments", {}), txn_no=txn)
            if self._enc[g["name"]]["curry"]:
                txn += 1
        return seq

    # ---- decoding ------------------------------------------------------------
    def decoder(self):
        return SeqDecoder(self.call_map)

    def decode_seq(self, calls):
        d = self.decoder()
        native = []
        for name, args in calls:
            ev = d.step(name, args)
            if ev is not None:
                native.append(ev)
        return native, d.flush()


class SeqDecoder:
    """Stateful fold over a variant call sequence -> native calls + flags.

    step() returns a native (name, args) when the call completes a native
    action (plain call, execute, commit), or None for meta calls
    (resolver / begin / set). Off-map names raise KeyError — callers
    (classify_call / SimEnv) decide policy for those.
    """

    def __init__(self, call_map):
        self.cmap = call_map
        self.txns = {}                             # txn_id -> {"op":…, "args":…, "done":bool}
        self._txn_no = 0
        self.flags = []                            # (flag, detail) pairs, in order

    def new_txn_id(self):
        self._txn_no += 1
        return f"{TXN_PREFIX}{self._txn_no}"

    def step(self, name, args):
        args = dict(args or {})
        e = self.cmap[name]
        kind = e.get("kind", "execute" if e.get("namespaced") else "named")
        if kind == "named":
            native = dict(e.get("fixed", {}))
            native.update(self._unpack(e, args))
            return e["op"], native
        if kind == "execute":
            err = validate_merged(e, args)
            if err:
                self.flags.append(("bad_execute", err))
            op, native = decode_merged(e, args)
            if e.get("ops") is not None and op not in e["ops"]:
                self.flags.append(("bad_operation", op))
            return op, native
        if kind == "pred":
            for prm, lo, hi in e.get("preds", []):
                v = args.get(prm)
                if not _in_interval(v, lo, hi):
                    self.flags.append(("pred_mismatch", f"{name}({prm}={v!r})"))
            native = dict(e.get("fixed", {}))
            native.update(self._unpack(e, args))
            return e["op"], native
        if kind == "resolver":
            return None                            # SimEnv answers; no native action
        if kind == "curry_begin":
            tid = self.new_txn_id()
            self.txns[tid] = {"op": e["op"], "args": {}, "done": False}
            self.last_txn_id = tid                 # SimEnv returns this to the model
            return None
        if kind == "curry_set":
            tid = args.pop("txn_id", None)
            t = self.txns.get(tid)
            if t is None or t["op"] != e["op"]:
                self.flags.append(("set_unknown_txn", f"{name}({tid})"))
            elif t["done"]:
                self.flags.append(("set_after_commit", f"{name}({tid})"))
            else:
                t["args"].update(self._resolve_handles(e, args))
            return None
        if kind == "curry_commit":
            tid = args.get("txn_id")
            t = self.txns.get(tid)
            if t is None or t["op"] != e["op"]:
                self.flags.append(("commit_unknown_txn", f"{name}({tid})"))
                return None
            if t["done"]:
                self.flags.append(("double_commit", f"{name}({tid})"))
                return None
            t["done"] = True
            missing = [r for r in e.get("required", []) if r not in t["args"]]
            if missing:
                self.flags.append(("commit_missing_required", f"{tid}:{missing}"))
            return e["op"], dict(t["args"])
        raise ValueError(f"unknown call_map kind {kind!r} for {name}")

    def _resolve_handles(self, e, args):
        """Swap *_ref handle values back to raw values (indirect); flag unknowns."""
        ind = e.get("ind_map") or {}
        out = {}
        for k, v in args.items():
            if k in ind:
                prm, _rname, table = ind[k]
                if v in table:
                    out[prm] = table[v]
                else:
                    self.flags.append(("bad_handle", f"{k}={v!r}"))
                    out[prm] = f"<UNRESOLVED:{v}>"
            else:
                out[k] = v
        return out

    def _unpack(self, e, args):
        """pack/lift/indirect reversal for a plain named call."""
        out = {}
        pack = e.get("pack_map") or {}
        lift = e.get("lift_map") or {}
        for k, v in self._resolve_handles(e, args).items():
            if k in pack and isinstance(v, dict):
                out.update(v)
            elif k in lift:
                out.setdefault(lift[k], {})
                if isinstance(out[lift[k]], dict):
                    out[lift[k]][k] = v
            else:
                out[k] = v
        return out

    def flush(self):
        for tid, t in self.txns.items():
            if not t["done"]:
                self.flags.append(("dangling_txn", f"{tid}:{t['op']}"))
        return list(self.flags)


def compile_spec(tools, spec, group=None):
    plan, merges = _norm_plan(tools, spec, group=group)
    byname = {t["name"]: t for t in tools}
    merged_set = {n for slot in merges.values() for n in slot["members"]}
    catalog, call_map, enc = [], {}, {}
    resolvers = []

    for t in tools:
        name = t["name"]
        if name in merged_set:
            continue
        p = plan[name]
        t2, pack_map, lift_map = _apply_pack_lift(t, p["pack"], p["lift"])
        t2, ind_map = _apply_indirect(t2, p["indirect"], resolvers, call_map)
        # surface ops last: resolver names/handle tables stay on native names
        if p["nodesc"]:
            t2 = _strip_desc(t2)
        if p["shuffle"] is not None:
            t2 = _shuffle_params(t2, p["shuffle"])
        if p["rename"]:
            t2 = {**t2, "name": p["rename"]}
        recipe = {"kind": "plain", "curry": p["curry"], "baked": list(p["baked"]),
                  "pred": {}, "pack_map": pack_map, "lift_map": lift_map,
                  "ind_map": ind_map, "base": t2["name"]}
        if p["curry"]:
            fns = _curry_tools(t2, call_map)
            for f in fns:
                e2 = call_map[f["name"]]
                e2["op"] = name                    # decode to NATIVE name (rename-safe)
                if f["name"].startswith("set_"):
                    e2["ind_map"] = ind_map
                if e2.get("required"):             # commit: required in NATIVE names
                    e2["required"] = [ind_map[r][0] if r in ind_map else r
                                      for r in e2["required"]]
            catalog += fns
            enc[name] = recipe
            continue
        # split_enum baking (cartesian) on the packed/indirected schema
        fns, cmap = _named_tools_mapped(t2, p["baked"])
        # predicate splits multiply each baked fn by the interval grid
        pred_grids = [(prm, _pred_intervals(m["cuts"], m["labels"]))
                      for prm, m in p["pred"].items()]
        out_fns, out_map = [], {}
        for f in fns:
            base_entry = cmap[f["name"]]
            combos = itertools.product(*[g for _, g in pred_grids]) if pred_grids else [()]
            for combo in combos:
                nm, desc = f["name"], f["description"]
                entry = dict(base_entry)
                props = {k: dict(v) for k, v in f["parameters"]["properties"].items()}
                for (prm, _g), (lab, lo, hi) in zip(pred_grids, combo):
                    nm += f"__{prm}_{lab}"
                    rng = (f"{prm} in [{lo if lo is not None else '-inf'}, "
                           f"{hi if hi is not None else 'inf'})")
                    desc = f"{desc} (handles {rng})".strip()
                    if lo is not None:
                        props[prm]["minimum"] = lo
                    if hi is not None:
                        props[prm]["exclusiveMaximum"] = hi
                    entry = {**entry, "kind": "pred",
                             "preds": entry.get("preds", []) + [(prm, lo, hi)]}
                if not pred_grids:
                    entry.setdefault("kind", "named")
                entry.update({"pack_map": pack_map, "lift_map": lift_map, "ind_map": ind_map})
                entry["op"] = name                 # decode to NATIVE name (rename-safe)
                out_fns.append({**f, "name": nm, "description": desc,
                                "parameters": {**f["parameters"], "properties": props}})
                out_map[nm] = entry
        catalog += out_fns
        call_map.update(out_map)
        recipe["pred"] = {prm: g for prm, g in pred_grids}
        enc[name] = recipe

    for into, slot in merges.items():
        members = slot["members"]
        encoding = slot["encoding"]
        ms = [byname[n] for n in members]
        if encoding == "nested":
            ex = execute_tool_nested(ms, name=into)
        elif encoding == "union":
            ex = execute_tool_union(ms, name=into)
        else:
            ex = _execute_tool(ms, name=into)
        catalog.append(ex)
        call_map[into] = {
            "kind": "execute",
            "encoding": encoding,
            "op": None,
            "fixed": {},
            "namespaced": encoding == "flat",
            "ops": [m["name"] for m in ms],
            "op_params": member_param_index(ms),
        }
        for m in members:
            enc[m] = {"kind": "merged", "entry": into, "encoding": encoding,
                      "curry": False, "baked": [],
                      "pred": {}, "pack_map": {}, "lift_map": {}, "ind_map": {}}

    catalog = resolvers + catalog
    return Compiled(catalog, call_map, enc)


def load_spec(s):
    """SCHEMA_SPEC env accepts inline JSON or a file path."""
    s = s.strip()
    if s.startswith("{"):
        return json.loads(s)
    return json.load(open(s, encoding="utf-8"))
