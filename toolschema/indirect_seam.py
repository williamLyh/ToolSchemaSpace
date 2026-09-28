"""Reference-indirection seam for real tau2 environments.

Activated by ``SCHEMA_TAU2_INDIRECT=1``. Selected grounding parameters are
replaced by opaque ``*_ref`` handles. Resolver calls are answered locally from
the values observed in tau2's pinned task golds; decoded native calls alone are
returned to the orchestrator.
"""
import difflib
import hashlib
import json
import os

from .operators import compile_spec

INSTRUCTION = (
    "Some tool arguments require opaque references. Call the corresponding "
    "`resolve_<tool>__<parameter>` function with the user-mentioned value first, "
    "then pass its returned `handle` in the tool's `<parameter>_ref` argument. "
    "Never pass the raw value where a reference is required."
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COMPILED = {}
_PREFERRED = ("_id", "id", "_number", "email", "phone", "username", "name")


def _canon(tools_schema):
    return [tool["function"] if "function" in tool else tool
            for tool in tools_schema]


def _task_values(path=None):
    path = path or os.environ.get(
        "SCHEMA_TAU2_TASKS",
        os.path.join(ROOT, "adapters", "tau2", "data", "tasks.json"),
    )
    seen = {}
    if not os.path.exists(path):
        return seen
    for task in json.load(open(path, encoding="utf-8")):
        for action in task.get("gold", {}).get("actions", []):
            for parameter, value in action.get("arguments", {}).items():
                if isinstance(value, (str, int, float, bool)):
                    seen.setdefault((action["name"], parameter), set()).add(value)
    return seen


def _candidate(parameter, definition):
    if definition.get("type") not in ("string", "integer", "number"):
        return False
    lowered = parameter.lower()
    return any(token in lowered for token in _PREFERRED)


def _build_spec(functions, observed):
    ops = []
    for function in functions:
        name = function["name"]
        properties = function.get("parameters", {}).get("properties", {})
        candidates = []
        for parameter, definition in properties.items():
            values = set(observed.get((name, parameter), set()))
            values.update(definition.get("enum", []))
            if len(values) < 2:
                continue
            priority = 0 if _candidate(parameter, definition) else 1
            candidates.append((priority, parameter, values))
        if not candidates:
            continue
        _, parameter, values = sorted(
            candidates, key=lambda item: (item[0], item[1]))[0]
        ops.append({
            "op": "indirect",
            "tool": name,
            "param": parameter,
            "values": sorted(values, key=str),
        })
    return {"ops": ops}


def _compiled_for(tools_schema, observed=None):
    functions = _canon(tools_schema)
    observed = _task_values() if observed is None else observed
    spec = _build_spec(functions, observed)
    signature = hashlib.sha1(json.dumps({
        "tools": sorted(function["name"] for function in functions),
        "spec": spec,
    }, sort_keys=True).encode()).hexdigest()
    if signature not in _COMPILED:
        _COMPILED[signature] = compile_spec(functions, spec)
    return _COMPILED[signature]


class IndirectSeam:
    MAX_META_ROUNDS = int(os.environ.get("SCHEMA_TAU2_MAXMETA", "8"))

    def __init__(self, compiled, control):
        self.comp = compiled
        self.cmap = compiled.call_map
        self.dec = compiled.decoder()
        self.control = control
        self.tools = [{"type": "function", "function": tool}
                      for tool in compiled.tools]
        self.decoded = []
        self.n_meta = 0
        self.n_rejected = 0

    @classmethod
    def get(cls, tools_schema, observed=None):
        return cls(
            _compiled_for(tools_schema, observed=observed),
            os.environ.get("SCHEMA_CONTROL", "hard"),
        )

    def _resolve(self, entry, args):
        query = str(args.get("query", "")).strip()
        table = entry.get("table", {})
        by_value = {str(value).lower(): handle
                    for handle, value in table.items()}
        if query.lower() in by_value:
            return json.dumps({"handle": by_value[query.lower()]})
        close = difflib.get_close_matches(
            query.lower(), list(by_value), n=3, cutoff=0.6)
        if len(close) == 1 or (
                close and difflib.SequenceMatcher(
                    None, query.lower(), close[0]).ratio() >= 0.85):
            return json.dumps({
                "handle": by_value[close[0]],
                "matched": table[by_value[close[0]]],
            })
        self.n_rejected += 1
        if close:
            return json.dumps({
                "error": "ambiguous or no exact match",
                "candidates": [table[by_value[value]] for value in close],
            })
        return json.dumps({
            "error": f"no {entry['param']} matching {query!r}",
        })

    def process(self, tool_calls):
        metas, natives = [], []
        for tool_call in tool_calls:
            name = tool_call.function.name
            raw_args = tool_call.function.arguments or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            entry = self.cmap.get(name)
            if entry is None:
                if self.control != "hard":
                    natives.append((tool_call.id, name, args))
                    continue
                self.n_rejected += 1
                metas.append((tool_call.id, name, raw_args, json.dumps({
                    "error": f"Unknown function `{name}`. Use only provided tools.",
                })))
                continue
            if entry.get("kind") == "resolver":
                self.n_meta += 1
                metas.append((
                    tool_call.id, name, raw_args, self._resolve(entry, args)))
                continue
            before = len(self.dec.flags)
            native = self.dec.step(name, args)
            new_flags = self.dec.flags[before:]
            if new_flags and self.control == "hard":
                self.n_rejected += 1
                metas.append((tool_call.id, name, raw_args, json.dumps({
                    "error": new_flags[-1][1],
                    "hint": "resolve the value first and use the returned handle",
                })))
            elif native is not None:
                natives.append((tool_call.id, native[0], native[1]))
        return metas, natives
