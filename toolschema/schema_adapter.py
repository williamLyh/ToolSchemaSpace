"""Runtime schema-variant adapter, imported by the three eval environments.

A benchmark run becomes *variant-aware* by setting:

    SCHEMA_VARIANT  = 0..9      (granularity index; 0 = coarse, 9 = fine extreme)
    SCHEMA_DATASET  = bfcl_multiturn | tau_bench | appworld   (optional; logging)

When `SCHEMA_VARIANT` is unset, `SchemaAdapter.from_env()` returns ``None`` and
every hooked environment falls back to native behavior — existing runs untouched.

Design: **live transformation.** Rather than read a pre-dumped catalog (which can
drift from what an env actually exposes — e.g. the tau_bench loader scraped an
older tool set than the tau2 runtime), the adapter transforms the environment's
*own native tool schemas* at the seam. This guarantees the variant's operation
set always matches the runtime exactly. Per group it computes

    level = round(k / (K-1) * (N + M))            # N ops, M bakeable enum params

and calls `schema_transform.group_variant_mapped` to get the forward `tools`
(model-facing) plus a `call_map` (to decode a variant call back to native).

Two usage shapes:
  - stateless `transform(native_openai_tools) -> (variant_tools, call_map)`
    when schema-build and call-decode happen in the same scope (tau2).
  - group-cached `variant_openai_tools(group, native)` + `decode(group, name,
    args)` when the two seams are in different functions (BFCL, AppWorld).

`*_openai_tools` speak the OpenAI tool shape
``{"type":"function","function":{name,description,parameters}}``.
"""

import os
import re

from .schema_transform import _bake_plan, decode_call, group_variant_mapped

K = int(os.environ.get("SCHEMA_K", "10"))    # ladder length: v0 coarsest, v5 native, v9 finest
NATIVE_K = K // 2                            # variant index that is ALWAYS native (v10 for K=21)

# Piecewise granularity mapping so a fixed index is native for every group,
# regardless of its (N ops, M bakeable enum params):
#   k in [0, NATIVE_K] : coarse -> native   (level 0 .. N, merge side)
#   k == NATIVE_K      : native              (level = N: all named, none merged/baked)
#   k in [NATIVE_K, 9] : native -> fine      (level N .. N+M, bake side)
# Groups with M==0 (no bakeable enums) simply have no fine side: k>=NATIVE_K all
# equal native. This makes v0=coarse, v5=native, v9=fine comparable across datasets.
def variant_level(k, N, M):
    if k <= NATIVE_K:
        return round(k / NATIVE_K * N) if NATIVE_K else N
    return N + round((k - NATIVE_K) / (K - 1 - NATIVE_K) * M)

# Prompt instruction injected (when a variant is active) telling the model to use
# ONLY the provided tools. Makes the schema binding visible to the model and is a
# prerequisite for fair hard-control scoring.
TOOL_INSTRUCTION = (
    "IMPORTANT: You are given a specific set of callable functions as your tools. "
    "You must accomplish the task using ONLY these provided functions and their "
    "declared arguments. Do not call any function that is not in the provided tool "
    "list, even if you know such a function exists. Encode every action using the "
    "tools exactly as defined."
)


def classify_call(call_map, control, name, args):
    """Decide what to do with a model call under a given control mode.

    Returns (action, native_name, native_args):
      - name IS an exposed variant tool  -> ("exec", decoded native name/args)
        …but under hard control, on-schema calls that violate the schema's own
        declared domain are rejected too: an `execute` whose `operation` is not
        in the dispatch enum, or a predicate-split function fed a value outside
        its interval (2026-07 audit: decode used to wave these through).
      - sequence-operator meta calls (resolver / begin / set) -> ("meta", ...)
        — they carry no native action; single-shot callers should ignore them
        (agentic callers use SeqDecoder / SimEnv instead of this shortcut).
      - name is NOT exposed (a native/off-schema name):
          control == "loose" -> ("exec", name, args)   # run it anyway (outcome)
          control == "hard"  -> ("reject", name, args)  # off-schema: do not run
    """
    if name in call_map:
        entry = call_map[name]
        kind = entry.get("kind")
        if kind in ("resolver", "curry_begin", "curry_set"):
            return "meta", name, dict(args or {})
        if kind == "curry_commit":
            return "meta", entry.get("op"), dict(args or {})
        nm, na = decode_call(call_map, name, args)
        if control == "hard":
            if entry.get("kind") == "execute" or entry.get("encoding") in ("nested", "union") \
                    or entry.get("namespaced"):
                from .merge_encodings import validate_merged
                if entry.get("ops") is not None and validate_merged(entry, args):
                    return "reject", nm, na
            if entry.get("namespaced") and entry.get("ops") is not None \
                    and nm not in entry["ops"]:
                return "reject", nm, na            # smuggled off-enum operation
            for prm, lo, hi in entry.get("preds", []):
                v = (args or {}).get(prm)
                ok = _num_in(v, lo, hi)
                if not ok:
                    return "reject", nm, na        # value outside declared interval
        return "exec", nm, na
    if control == "hard":
        return "reject", name, args
    return "exec", name, args


def _num_in(v, lo, hi):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return False
    return (lo is None or v >= lo) and (hi is None or v < hi)


# --------------------------------------------------------------------------- #
# Tier-A enum lifting (self-contained copy of loaders/common.py, stdlib-only).
# A `string` param is *bakeable into a function name* only when its description
# names a CLOSED value set. High-precision: never promotes open / grounding
# strings. Needed at runtime for envs (e.g. BFCL) whose live tool docs carry the
# value set only in prose, not as a structured `enum`. Envs that already emit
# structured enums (tau2 Literals, AppWorld) are unaffected (detect returns None
# when `enum` is already present).
# --------------------------------------------------------------------------- #
_OPEN = re.compile(r"\betc\b|\.\.\.|e\.g\.|such as|for example|for instance", re.I)
_QUOTED = re.compile(r"""["']([^"']+)["']""")
_POSSIBLE = re.compile(r"possible values[^.]*", re.I)
_OPTIONS = re.compile(r"options?\b[^:.]{0,15}:\s*([^.]+)", re.I)
_PAREN = re.compile(r"\(([^)]*)\)")


def _split_set(s):
    out = []
    for p in re.split(r"/|,|\bor\b|\band\b", s, flags=re.I):
        p = p.strip().strip("'\"").strip()
        if p and len(p) <= 20 and " " not in p:
            out.append(p)
    return out


def _valid_tokens(vals):
    out, seen = [], set()
    for v in vals:
        v = v.strip().strip("'\"").strip()
        if not v:
            continue
        if not re.search(r"[A-Za-z0-9]", v) or len(v) > 30:
            return None
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out if len(out) >= 2 else None


def _detect_enum(pdef):
    if pdef.get("type") != "string" or "enum" in pdef:
        return None
    desc = pdef.get("description", "")
    if _OPEN.search(desc):
        return None
    m = _POSSIBLE.search(desc)
    if m:
        v = _valid_tokens(_QUOTED.findall(m.group(0)))
        if v:
            return v
    m = _OPTIONS.search(desc)
    if m:
        v = _valid_tokens(_split_set(m.group(1)))
        if v:
            return v
    for body in _PAREN.findall(desc):
        v = _valid_tokens(_QUOTED.findall(body))
        if v:
            return v
        if re.search(r"/|\bor\b", body, re.I):
            v = _valid_tokens(_split_set(body))
            if v:
                return v
    return None


def _lift_enum(pdef):
    if not isinstance(pdef, dict):
        return pdef
    vals = _detect_enum(pdef)
    return {**pdef, "enum": vals} if vals else pdef


# --------------------------------------------------------------------------- #
# OpenAI tool-schema <-> canonical 3-key schema
# --------------------------------------------------------------------------- #
def _resolve_prop(pdef, defs):
    """Flatten a JSON-schema property so `enum`/`type` are visible to the
    transformer. Resolves ``$ref`` into ``$defs``, merges ``allOf``, and for
    ``anyOf`` (e.g. Optional[X]) keeps the first non-null branch. Best-effort:
    unknown shapes pass through unchanged."""
    if not isinstance(pdef, dict):
        return pdef
    out = dict(pdef)
    ref = out.pop("$ref", None)
    if ref and ref.startswith("#/$defs/"):
        out = {**defs.get(ref.split("/")[-1], {}), **out}
    if "allOf" in out:
        merged = {}
        for sub in out.pop("allOf"):
            merged.update(_resolve_prop(sub, defs))
        out = {**merged, **out}
    if "anyOf" in out:
        branches = [b for b in out.pop("anyOf")
                    if not (isinstance(b, dict) and b.get("type") == "null")]
        if branches:
            out = {**_resolve_prop(branches[0], defs), **out}
    return out


def _canonicalize(openai_tool):
    """OpenAI tool dict -> canonical {name, description, parameters{type,properties,required}}."""
    fn = openai_tool.get("function", openai_tool)
    params = fn.get("parameters") or {}
    defs = params.get("$defs", {}) or {}
    props = {p: _lift_enum(_resolve_prop(pd, defs))
             for p, pd in (params.get("properties") or {}).items()}
    return {
        "name": fn["name"],
        "description": fn.get("description", "") or "",
        "parameters": {"type": "object", "properties": props,
                       "required": list(params.get("required", []) or [])},
    }


def _to_openai(canon):
    """Canonical schema -> OpenAI tool dict; strips internal `optional` markers."""
    params = canon["parameters"]
    props = {p: {k: v for k, v in pd.items() if k != "optional"}
             for p, pd in params.get("properties", {}).items()}
    return {"type": "function", "function": {
        "name": canon["name"], "description": canon.get("description", ""),
        "parameters": {"type": "object", "properties": props,
                       "required": list(params.get("required", []))}}}


class SchemaAdapter:
    INSTRUCTION = TOOL_INSTRUCTION

    def __init__(self, k=None, dataset=None, control=None, spec=None):
        self.k = int(k) if k is not None else None
        self.dataset = dataset
        self.control = (control or os.environ.get("SCHEMA_CONTROL") or "loose").lower()
        self.spec = spec                      # operator spec dict (overrides k-ladder)
        self._cache = {}                      # group -> {"tools": [...], "call_map": {...}, ...}
        assert self.k is not None or self.spec is not None, "need SCHEMA_VARIANT or SCHEMA_SPEC"

    # -- construction -------------------------------------------------------
    _cached = False
    _instance = None

    @classmethod
    def from_env(cls):
        if cls._cached:
            return cls._instance
        k = os.environ.get("SCHEMA_VARIANT")
        spec_s = os.environ.get("SCHEMA_SPEC")
        if spec_s:
            from .operators import load_spec
            inst = cls(k=int(k) if k not in (None, "") else None,
                       dataset=os.environ.get("SCHEMA_DATASET"), spec=load_spec(spec_s))
        elif k not in (None, ""):
            inst = cls(int(k), os.environ.get("SCHEMA_DATASET"))
        else:
            inst = None
        cls._cached, cls._instance = True, inst
        return inst

    def classify(self, group, name, args):
        """Group-cached form of `classify_call` (BFCL/AppWorld use the stateless one)."""
        return classify_call(self.call_map_for(group), self.control, name, args)

    # -- core ---------------------------------------------------------------
    def transform(self, native_openai_tools, group=None):
        """Stateless: native OpenAI tools -> (variant OpenAI tools, call_map).

        With a spec (SCHEMA_SPEC), compiles the operator spec; otherwise uses
        the k-ladder piecewise mapping so self.k==NATIVE_K is exactly native.
        `group` disambiguates group-stamped ops in multi-group spec files
        (tool names are not unique across groups).
        """
        canon = [_canonicalize(t) for t in native_openai_tools]
        if self.spec is not None:
            from .operators import compile_spec
            comp = compile_spec(canon, self.spec, group=group)
            self.last_compiled = comp          # agentic callers reuse encoder/decoder
            return [_to_openai(t) for t in comp.tools], comp.call_map
        N = len(canon)
        M = len(_bake_plan(canon))
        gv = group_variant_mapped(canon, variant_level(self.k, N, M))
        return [_to_openai(t) for t in gv["tools"]], gv["call_map"]

    # -- group-cached convenience (BFCL / AppWorld) -------------------------
    def variant_openai_tools(self, group, native_openai_tools):
        """Transform `native_openai_tools` for `group`, cache the call_map, and
        return the variant OpenAI tools. Idempotent per group."""
        if group not in self._cache:
            tools, cmap = self.transform(native_openai_tools, group=group)
            self._cache[group] = {"tools": tools, "call_map": cmap}
        return self._cache[group]["tools"]

    def call_map_for(self, group):
        c = self._cache.get(group)
        return c["call_map"] if c else {}

    def decode(self, group, name, args):
        """Variant-vocabulary call -> native (name, args). Pass-through if unmapped."""
        return decode_call(self.call_map_for(group), name, args)
