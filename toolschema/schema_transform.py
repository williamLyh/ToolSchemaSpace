"""Schema-granularity transformer.

Rewrite a BFCL single-call sample (task + schema + gold) into a functionally equivalent
schema of a different granularity. Core abstraction: an action consists of "decision fields"
(which base function + each argument value); a schema partitions these fields into
  - name-encoded: baked into the function name (one function per value combination)
  - arg-encoded : kept as arguments
Granularity = how many fields are pushed into the function name.

This file implements the cleanest, fully deterministic end first: coarsen -- collapse all
candidate functions into one generic `execute(operation, <namespaced params>)`, with every decision in the arguments.
This is the "coarse extreme" that native BFCL data lacks entirely.

Each variant also rewrites the gold so it expresses the same underlying action in the variant's
vocabulary, and can be scored directly by checker.check_call.
"""

import itertools
import re
from collections import defaultdict

from .checker import check_call


def _nskey(fn, param):
    """Namespace parameters by their owning function to avoid name clashes after merging."""
    return f"{fn}::{param}"


def coarsen(rec):
    """All candidate functions -> one generic execute(operation, <namespaced params>).

    - operation: enum over the original candidate names; required (no '' omission).
    - every original parameter is namespaced as `fn::param` and marked optional in the schema.
    - gold rewrite: operation=gold_name, gold arguments move under namespaced keys (keeping the
      acceptable-value lists); every other namespaced parameter is set to [''] (must be omitted).
      Note: requiredness is still enforced by gold_args (a required parameter's acceptable values exclude ''); the schema
      is only relaxed -- loose schema, strict gold check, which is what we want.
    """
    tools = rec["tools"]
    props = {
        "operation": {
            "type": "string",
            "enum": [t["name"] for t in tools],
            "description": "Which operation to perform.",
        }
    }
    for t in tools:
        for pname, pdef in t["parameters"]["properties"].items():
            d = dict(pdef)
            d["optional"] = True
            d["description"] = f"[{t['name']}] " + d.get("description", "")
            props[_nskey(t["name"], pname)] = d

    schema = {
        "name": "execute",
        "description": "Universal entry point. Pick `operation`, then supply the "
                       "namespaced arguments belonging to that operation.",
        "parameters": {"type": "dict", "properties": props, "required": ["operation"]},
    }

    gold_args = {"operation": [rec["gold_name"]]}
    for p, acc in rec["gold_args"].items():
        gold_args[_nskey(rec["gold_name"], p)] = acc
    for key in props:
        if key != "operation" and key not in gold_args:
            gold_args[key] = [""]          # parameters of other operations: must be omitted

    return {
        "id": rec["id"],
        "category": rec["category"] + "/coarse",
        "query": rec["query"],
        "tools": [schema],
        "gold_name": "execute",
        "gold_args": gold_args,
        # name/arg split of the decision fields (for the entropy measures): the coarse end has 0 name-side dims.
        "name_fields": [],
        "arg_fields": ["operation"] + [k for k in gold_args if k != "operation"],
    }


def native(rec):
    """Identity transform (BFCL native granularity), with name/arg field labels added for uniform handling.

    name side: choice of base function (carries entropy only when >1 candidate); arg side: gold arguments.
    """
    return {
        **rec,
        "category": rec["category"] + "/native",
        "name_fields": ["operation"],            # which function = carried by the name
        "arg_fields": list(rec["gold_args"].keys()),
    }


# Variant registry; register further transforms here.
VARIANTS = {"native": native, "coarse": coarsen}


def make_variants(rec, which=VARIANTS):
    return {name: fn(rec) for name, fn in which.items()}


# ===========================================================================
# Group-level granularity sweep -- transforms the schema only, never the gold.
#
# Input: one group's tool catalog (list of canonical tool dicts, each with
#   name / description / parameters{type, properties, required})。
# Output: a sequence of functionally equivalent catalogs of increasing granularity, spanning
#   level 0     = fewest functions (one generic `execute(operation, ...)`)
#   …
#   level N+M   = fewest parameters (every enumerable field baked into names)
#
# One integer level walks a two-segment path:
#   Segment A (coarse -> native): peel base operations out of execute one by one as named functions.
#       level 0 -> 1 function; level k -> k named functions + execute(remaining N-k);
#       level N -> native (N named functions, no execute).
#   Segment B (native -> fine): bake each operation's *enumerable* parameters (enum / boolean) into
#       the name one by one; parameters shrink and functions multiply up to the fine extreme.
#
# Constraint: only enumerable fields can enter function names; continuous / grounding parameters stay
# as arguments, so the fine extreme is bounded.
# ===========================================================================

def _is_enumerable(pdef):
    return "enum" in pdef or pdef.get("type") == "boolean"


def _enum_values(pdef):
    if "enum" in pdef:
        return list(pdef["enum"])
    if pdef.get("type") == "boolean":
        return [True, False]
    return []


def _gprops(tool):
    return tool["parameters"].get("properties", {})


def _greq(tool):
    return list(tool["parameters"].get("required", []))


def _san(v):
    """Normalize an enum value into a string that is safe as a function-name suffix."""
    if isinstance(v, bool):
        return "true" if v else "false"
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(v)).strip("_").lower()
    return s or "x"


def _peel_order(tools):
    """Segment A peel order: operations with the most parameters first (sheds execute's parameter bloat fastest)."""
    return [t["name"] for t in sorted(tools, key=lambda t: (-len(_gprops(t)), t["name"]))]


def _bake_plan(tools):
    """Segment B baking order: all (operation, enumerable parameter) pairs by ascending cardinality (function count grows gently)."""
    flat = []
    for t in tools:
        for p, pd in _gprops(t).items():
            if _is_enumerable(pd):
                flat.append(((t["name"], p), len(_enum_values(pd))))
    return [op_p for op_p, _ in sorted(flat, key=lambda x: (x[1], x[0]))]


def _execute_tool(merged, name="execute"):
    """Collapse the `merged` operations into one execute(operation, <namespaced params>).

    By default the description lists only the N operations it carries (in a mixed schema a "Universal entry
    point" description overreaches). SCHEMA_LEGACY_EXEC_DESC=1 restores the generic description,
    which some earlier runs used.
    """
    import os
    props = {
        "operation": {
            "type": "string",
            "enum": [t["name"] for t in merged],
            "description": "Which operation to perform.",
        }
    }
    for t in merged:
        for pn, pd in _gprops(t).items():
            d = dict(pd)
            d["optional"] = True
            d["description"] = f"[{t['name']}] " + d.get("description", "")
            props[_nskey(t["name"], pn)] = d
    ops = ", ".join(t["name"] for t in merged)
    if os.environ.get("SCHEMA_LEGACY_EXEC_DESC"):
        desc = ("Universal entry point. Pick `operation`, then supply the "
                "namespaced arguments belonging to that operation.")
    else:
        desc = (f"Dispatch entry point handling ONLY these {len(merged)} operations: "
                f"{ops}. Pick `operation`, then supply that operation's namespaced "
                f"arguments.")
    return {
        "name": name,
        "description": desc,
        "parameters": {"type": "object", "properties": props, "required": ["operation"]},
    }


def _named_tools_mapped(tool, baked):
    """Expand one operation into named functions (as _named_tools), with the reverse map.

    Returns (fns, cmap): fns is the list of named-function schemas; cmap[fn_name] =
    {"op": native op name, "fixed": {baked param: value}, "namespaced": False}.
    `fixed` is captured at generation time (combo) rather than re-parsed from the lossy _san name suffix.
    """
    grids = [_enum_values(_gprops(tool)[p]) for p in baked]
    rest_props = {pn: dict(pd) for pn, pd in _gprops(tool).items() if pn not in baked}
    rest_req = [r for r in _greq(tool) if r not in baked]
    out, cmap = [], {}
    for combo in (itertools.product(*grids) if baked else [()]):
        suffix = "".join(f"__{p}_{_san(v)}" for p, v in zip(baked, combo))
        desc = tool.get("description", "")
        if baked:
            fixed = ", ".join(f"{p}={v}" for p, v in zip(baked, combo))
            desc = f"{desc} (fixed: {fixed})".strip()
        name = tool["name"] + suffix
        out.append({
            "name": name,
            "description": desc,
            "parameters": {"type": "object",
                           "properties": dict(rest_props),
                           "required": list(rest_req)},
        })
        cmap[name] = {"op": tool["name"],
                      "fixed": {p: v for p, v in zip(baked, combo)},
                      "namespaced": False}
    return out, cmap


def _named_tools(tool, baked):
    """Expand one operation into named functions: enumerable parameters in `baked` go into the name (Cartesian
    product), the rest stay as arguments. Empty `baked` => the native form (a single function)."""
    fns, _ = _named_tools_mapped(tool, baked)
    return fns


def level_partition(tools, level):
    """Name/arg split of the decision fields for a given level (shared by group_variant and metrics).

    Returns (named, merged, baked_by_op, N, M):
      named       : operations whose identity is encoded in a function name
      merged      : operation dicts still collapsed into execute (may be empty)
      baked_by_op : {op: [enumerable params baked into the name]}
      N, M        : number of operations / of bakeable parameters (level max = N+M)
    """
    peel = _peel_order(tools); N = len(peel)
    bake = _bake_plan(tools);  M = len(bake)
    level = max(0, min(int(level), N + M))
    baked_by_op = defaultdict(list)
    if level <= N:
        named = set(peel[:level])
        merged = [t for t in tools if t["name"] not in named]
        if len(merged) == 1:          # an execute over one operation degenerates to enum=1: treat it as named
            named.add(merged[0]["name"]); merged = []
    else:
        named = {t["name"] for t in tools}
        merged = []
        for op, p in bake[: level - N]:
            baked_by_op[op].append(p)
    return named, merged, baked_by_op, N, M


def group_variant(tools, level):
    """Schema variant of a group at a given level (tools catalog and stats only, no gold).

    level in [0, N+M]: 0 = fewest functions (single execute), N = native, N+M = fewest parameters (fine extreme).
    """
    named, merged, baked_by_op, N, M = level_partition(tools, level)

    catalog = []
    for t in tools:
        if t["name"] in named:
            catalog += _named_tools(t, baked_by_op.get(t["name"], []))
    if merged:
        catalog.append(_execute_tool(merged))
    name_ops = len(named)
    baked_params = sum(len(v) for v in baked_by_op.values())

    arg_counts = [len(_gprops(t)) for t in catalog]
    return {
        "level": min(max(0, int(level)), N + M),
        "max_level": N + M,
        "tools": catalog,
        "n_functions": len(catalog),
        "name_ops": name_ops,             # how many base operations have their identity in a name
        "baked_params": baked_params,     # how many enumerable parameters are baked into names
        "args_total": sum(arg_counts),
        "args_mean": (sum(arg_counts) / len(arg_counts)) if arg_counts else 0.0,
        "args_max": max(arg_counts) if arg_counts else 0,
    }


def group_sweep(tools):
    """Full sweep: level 0 -> N+M, one variant per level."""
    N = len(tools)
    M = len(_bake_plan(tools))
    return [group_variant(tools, lv) for lv in range(N + M + 1)]


# ===========================================================================
# Group-level variants with a reverse map (for evaluation injected into real environments, scored natively).
#
# group_variant only gives the forward schema (what the model sees). To turn calls the model makes in the
# variant vocabulary back into native calls (for the original executor / scorer) we need a call_map:
#   call_map[variant function name] = {
#       "op":         native operation name (None for an execute entry; decided by `operation` at runtime),
#       "fixed":      parameters baked into the name and their values (named/baked forms),
#       "namespaced": whether this is an execute entry (parameters carry an `op::param` namespace),
#       "ops":        (execute only) which native operations the entry collapses,
#   }
# decode_call uses this map to decode (variant name, variant args) into (native name, native args).
# ===========================================================================

def group_variant_mapped(tools, level):
    """Same as group_variant, but also returns the call_map (needed for reverse decoding)."""
    named, merged, baked_by_op, N, M = level_partition(tools, level)
    catalog, call_map = [], {}
    for t in tools:
        if t["name"] in named:
            fns, cmap = _named_tools_mapped(t, baked_by_op.get(t["name"], []))
            catalog += fns
            call_map.update(cmap)
    if merged:
        catalog.append(_execute_tool(merged))
        call_map["execute"] = {"op": None, "fixed": {}, "namespaced": True,
                               "ops": [t["name"] for t in merged]}
    return {
        "level": min(max(0, int(level)), N + M),
        "max_level": N + M,
        "tools": catalog,
        "call_map": call_map,
    }


def decode_call(call_map, name, args):
    """Decode one call (name, args) in the variant vocabulary into native (native_name, native_args).

    - execute entry (namespaced): native_name = args['operation']; strip the `op::` namespace prefix
      from the other arguments and drop omitted (''/None) ones.
    - named/baked: native_name = entry['op']; native_args = fixed | passed arguments.
    - unregistered names pass through unchanged (for the native variant or pass-through tools).
    """
    args = dict(args or {})
    entry = call_map.get(name)
    if entry is None:
        return name, args
    if entry.get("encoding") in ("nested", "union"):
        from .merge_encodings import decode_merged
        native_name, native_args = decode_merged(entry, args)
        return native_name, native_args
    if entry.get("namespaced"):
        native_name = args.get("operation")
        native_args = {}
        for k, v in args.items():
            if k == "operation":
                continue
            if v == "" or v is None:          # an explicitly omitted namespaced parameter
                continue
            p = k.split("::", 1)[1] if "::" in k else k
            native_args[p] = v
        return native_name, native_args
    native_args = dict(entry.get("fixed", {}))
    # arg_lower / arg_lift (nested args): reverse the packing recorded by the spec compiler
    # (same rule as operators.SeqDecoder._unpack, so single-call decoders agree with the
    # sequence decoder; the tau2 seam decodes through classify_call -> decode_call).
    pack = entry.get("pack_map") or {}
    lift = entry.get("lift_map") or {}
    for k, v in args.items():
        if k in pack and isinstance(v, dict):
            native_args.update(v)
        elif k in lift:
            native_args.setdefault(lift[k], {})
            if isinstance(native_args[lift[k]], dict):
                native_args[lift[k]][k] = v
        else:
            native_args[k] = v
    return entry["op"], native_args


def _canon_pred(gold_args):
    """Build a "correct prediction" from a variant's gold: each parameter takes its first non-empty acceptable
    value; parameters that only allow '' are omitted. Used for the round-trip self-check."""
    pred = {}
    for p, acc in gold_args.items():
        vals = [v for v in acc if v != ""]
        if vals:
            pred[p] = vals[0]
    return pred


def validate(rec_variant):
    """A variant's gold must be judged correct by its own checker (round-trip consistency)."""
    pred = _canon_pred(rec_variant["gold_args"])
    return check_call(rec_variant["gold_name"], pred,
                      rec_variant["gold_name"], rec_variant["gold_args"])
