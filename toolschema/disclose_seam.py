"""Progressive-disclosure seam for real tau2 environments.

Activated by ``SCHEMA_TAU2_DISCLOSE=1``. The model initially sees only
``list_methods`` and ``invoke``. ``list_methods`` is answered locally with the
domain's native method schemas; successful ``invoke`` calls are decoded to the
native tool call consumed by tau2, so execution and scoring remain unchanged.
"""
import hashlib
import json
import os

from .hierarchy import HierEnv, build_arm

INSTRUCTION = (
    "Tool argument schemas are progressively disclosed. First call "
    "`list_methods` for the available class, then call `invoke` using the exact "
    "method name and argument schema returned by that result."
)

_COMPILED = {}


def _canon(tools_schema, domain):
    rows = []
    for tool in tools_schema:
        fn = tool["function"] if "function" in tool else tool
        rows.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {
                "type": "object", "properties": {}, "required": []}),
            "group": domain,
        })
    return rows


def _compiled_for(tools_schema, domain):
    rows = _canon(tools_schema, domain)
    signature = hashlib.sha1(json.dumps(
        [domain, sorted(row["name"] for row in rows)]).encode()).hexdigest()
    if signature not in _COMPILED:
        _COMPILED[signature] = build_arm("disclose", rows)
    return _COMPILED[signature]


class DiscloseSeam:
    MAX_META_ROUNDS = int(os.environ.get("SCHEMA_TAU2_MAXMETA", "8"))

    def __init__(self, comp, domain, control):
        self.comp = comp
        self.domain = domain
        self.control = control
        self.env = HierEnv(comp, control=control)
        self.tools = [{"type": "function", "function": tool}
                      for tool in comp.tools]
        self.decoded = []
        self.n_meta = 0
        self.n_rejected = 0

    @classmethod
    def get(cls, tools_schema, domain=None):
        domain = domain or os.environ.get("SCHEMA_TAU2_DOMAIN", "tau2")
        return cls(
            _compiled_for(tools_schema, domain),
            domain,
            os.environ.get("SCHEMA_CONTROL", "hard"),
        )

    def process(self, tool_calls):
        """Return locally answered meta calls and decoded native calls."""
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
            before = len(self.env.native_actions)
            result = self.env.call(name, args)
            self.n_rejected = self.env.n_rejected
            if len(self.env.native_actions) > before:
                qualified, native_args = self.env.native_actions[-1]
                prefix = f"{self.domain}."
                native_name = (qualified[len(prefix):]
                               if qualified.startswith(prefix) else qualified)
                natives.append((tool_call.id, native_name, native_args))
                continue
            self.n_meta += 1
            metas.append((tool_call.id, name, raw_args, result))
        return metas, natives
