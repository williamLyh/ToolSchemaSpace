"""Hierarchy (class layer) over the WHOLE synthetic catalog.

`group`/`flatten` — one N-way selection vs a two-stage
hierarchical decision, with three surface realizations of the SAME logical
hierarchy, measured against a flat token-matched control. The class
layer is the domain (12 classes × ~14 methods = 168 tools); the model always
sees the FULL catalog, so cross-domain name collisions (set_volume etc.) make
class information load-bearing.

Arms (build_arm(arm, catalog)):

    flat        168 tools, original names (the 5 colliding pairs are renamed
                `<name>_<class>`); every description is prefixed "[<class>] "
                so class information is present as tokens — the token-matched
                flat control.
    qualified   (a) class.method naming: one tool per method, named
                `<class>__<method>`; descriptions unprefixed (the class moved
                from description tokens into the name).
    dispatch    (b) call(class, method, args): one tool per CLASS, named
                `<class>`, with an `operation` enum over the class's methods
                and namespaced arguments (schema_transform._execute_tool —
                the same surface as the ladder's merge, applied per class).
    disclose    (c) two-step progressive disclosure: `list_methods(class)`
                returns the class's full method schemas as a tool RESULT, and
                `invoke(class, method, arguments)` executes one; method
                argument schemas are NOT in the upfront catalog.

Native identity is CLASS-QUALIFIED: every decode yields ("<class>.<method>",
args) and golds are compared in that space — picking the right method in the
wrong class is wrong (that is the point of the collision pairs).

Equivalence is verified, never assumed: tests/test_hierarchy.py runs
every gold episode through encode_episode -> HierEnv and requires exact = 1.0
with zero rejections for all four arms.
"""
import json

from .schema_transform import _execute_tool

ARMS = ("flat", "qualified", "dispatch", "disclose")


def _canon(t):
    return {"name": t["name"], "description": t["description"],
            "parameters": t["parameters"]}


class HierCompiled:
    def __init__(self, arm, tools, call_map, enc_names):
        self.arm = arm
        self.tools = tools                     # canonical [{name,description,parameters}]
        self.call_map = call_map               # variant fn name -> entry
        self._enc = enc_names                  # (class, method) -> variant fn name (flat/qualified)

    # ---- oracle ---------------------------------------------------------- #
    def encode_episode(self, cls, gold_calls):
        seq = []
        if self.arm == "disclose":
            seq.append(("list_methods", {"class": cls}))
        for g in gold_calls:
            m, args = g["name"], dict(g.get("arguments", {}))
            if self.arm == "dispatch":
                vargs = {"operation": m}
                vargs.update({f"{m}::{k}": v for k, v in args.items()})
                seq.append((cls, vargs))
            elif self.arm == "disclose":
                seq.append(("invoke", {"class": cls, "method": m, "arguments": args}))
            else:
                seq.append((self._enc[(cls, m)], args))
        return seq


class HierEnv:
    """Per-episode executable env for the hierarchy arms (SimEnv analogue;
    stateless beyond the executed-action log — no transactions here).

    hard control rejects schema-domain violations with an error the model may
    recover from (unknown function / off-enum operation / unknown class or
    method); loose executes everything as given."""

    def __init__(self, comp, control="hard"):
        self.comp = comp
        self.cmap = comp.call_map
        self.control = control
        self.native_actions = []               # ("<class>.<method>", args)
        self.n_calls = 0
        self.n_rejected = 0

    def _reject(self, msg):
        self.n_rejected += 1
        return json.dumps({"error": msg})

    def call(self, name, args):
        self.n_calls += 1
        args = dict(args) if isinstance(args, dict) else {}
        e = self.cmap.get(name)
        if e is None:
            if self.control == "hard":
                return self._reject(f"Unknown function `{name}`. Use only the provided tools.")
            self.native_actions.append((name, args))
            return json.dumps({"status": "ok"})
        kind = e["kind"]

        if kind == "hier_named":
            self.native_actions.append((f"{e['cls']}.{e['op']}", args))
            return json.dumps({"status": "ok", "executed": e["op"]})

        if kind == "hier_execute":
            op = args.pop("operation", None)
            if op not in e["ops"]:
                if self.control == "hard":
                    return self._reject(f"operation must be one of {e['ops']}")
                op = op or "<missing_operation>"
            native = {}
            for k, v in args.items():
                if v in ("", None):
                    continue
                p = k.split("::", 1)[1] if "::" in k else k
                native[p] = v
            self.native_actions.append((f"{e['cls']}.{op}", native))
            return json.dumps({"status": "ok", "executed": op})

        if kind == "hier_list":
            cls = args.get("class")
            methods = e["classes"].get(cls)
            if methods is None:
                return self._reject(f"unknown class {cls!r}; classes: {sorted(e['classes'])}")
            return json.dumps({"class": cls, "methods": methods})

        if kind == "hier_invoke":
            cls, m = args.get("class"), args.get("method")
            margs = args.get("arguments")
            margs = dict(margs) if isinstance(margs, dict) else {}
            known = e["classes"].get(cls)
            if self.control == "hard":
                if known is None:
                    return self._reject(f"unknown class {cls!r}; classes: {sorted(e['classes'])}")
                if m not in known:
                    return self._reject(f"class `{cls}` has no method {m!r}; "
                                        f"call list_methods first. Methods: {known}")
            self.native_actions.append((f"{cls}.{m}", margs))
            return json.dumps({"status": "ok", "executed": m})

        raise ValueError(f"unknown kind {kind!r} for {name}")

    def finish(self):
        return []                              # no cross-call state to flush


def build_arm(arm, catalog):
    """catalog: tools.json rows [{name, description, parameters, group}]."""
    assert arm in ARMS, f"arm must be one of {ARMS}"
    from collections import Counter, defaultdict
    dup = {n for n, c in Counter(t["name"] for t in catalog).items() if c > 1}
    by_cls = defaultdict(list)
    for t in catalog:
        by_cls[t["group"]].append(t)

    tools, cmap, enc = [], {}, {}

    if arm in ("flat", "qualified"):
        for cls in sorted(by_cls):
            for t in by_cls[cls]:
                c = _canon(t)
                if arm == "flat":
                    nm = f"{t['name']}_{cls}" if t["name"] in dup else t["name"]
                    c["name"] = nm
                    c["description"] = f"[{cls}] {c['description']}"
                else:
                    nm = f"{cls}__{t['name']}"
                    c["name"] = nm
                tools.append(c)
                cmap[nm] = {"kind": "hier_named", "cls": cls, "op": t["name"]}
                enc[(cls, t["name"])] = nm
        return HierCompiled(arm, tools, cmap, enc)

    if arm == "dispatch":
        for cls in sorted(by_cls):
            ms = [_canon(t) for t in by_cls[cls]]
            ex = _execute_tool(ms, name=cls)
            tools.append(ex)
            cmap[cls] = {"kind": "hier_execute", "cls": cls,
                         "ops": [m["name"] for m in ms]}
        return HierCompiled(arm, tools, cmap, enc)

    # disclose
    classes = {cls: [t["name"] for t in by_cls[cls]] for cls in sorted(by_cls)}
    schemas = {cls: [_canon(t) for t in by_cls[cls]] for cls in sorted(by_cls)}
    class_lines = "; ".join(f"{cls}: {', '.join(classes[cls])}" for cls in sorted(classes))
    tools.append({
        "name": "list_methods",
        "description": ("List a class's methods with their FULL argument schemas. "
                        "Call this first: `invoke` requires the exact argument names, "
                        "which are only available here."),
        "parameters": {"type": "object",
                       "properties": {"class": {"type": "string",
                                                "enum": sorted(classes),
                                                "description": "The class to inspect."}},
                       "required": ["class"]}})
    cmap["list_methods"] = {"kind": "hier_list", "classes": schemas}
    tools.append({
        "name": "invoke",
        "description": (f"Invoke one method of one class. Available methods — {class_lines}. "
                        f"Argument names/types come from `list_methods`."),
        "parameters": {"type": "object",
                       "properties": {
                           "class": {"type": "string", "enum": sorted(classes),
                                     "description": "The class owning the method."},
                           "method": {"type": "string",
                                      "description": "Method name within the class."},
                           "arguments": {"type": "object",
                                         "description": "The method's arguments, per list_methods."}},
                       "required": ["class", "method", "arguments"]}})
    cmap["invoke"] = {"kind": "hier_invoke", "classes": classes}
    return HierCompiled(arm, tools, cmap, enc)
