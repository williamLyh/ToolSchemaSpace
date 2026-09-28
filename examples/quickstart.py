"""Quickstart: rewrite a native tool schema into variants and decode calls back.

Runs offline, with no model:  python examples/quickstart.py
"""
import json

from toolschema.operators import compile_spec
from toolschema.schema_adapter import SchemaAdapter

# A native catalog in the canonical shape {name, description, parameters}.
TOOLS = [
    {"name": "set_window", "description": "Open, close, or vent a car window.",
     "parameters": {"type": "object",
                    "properties": {"position": {"type": "string", "enum": ["front_left", "front_right"]},
                                   "state": {"type": "string", "enum": ["open", "closed", "vent"]}},
                    "required": ["position", "state"]}},
    {"name": "set_seat_heat", "description": "Set the heating level of a seat.",
     "parameters": {"type": "object",
                    "properties": {"seat": {"type": "string"}, "level": {"type": "integer"}},
                    "required": ["seat", "level"]}},
]
GOLD = [{"name": "set_window", "arguments": {"position": "front_left", "state": "vent"}},
        {"name": "set_seat_heat", "arguments": {"seat": "driver", "level": 2}}]

SPECS = {
    "merge (nested)": {"ops": [{"op": "merge", "tools": ["set_window", "set_seat_heat"],
                                "into": "execute", "encoding": "nested"}]},
    "split (enum)": {"ops": [{"op": "split_enum", "tool": "set_window", "params": ["state"]}]},
    "transaction": {"ops": [{"op": "curry", "tool": "set_window"}, {"op": "curry", "tool": "set_seat_heat"}]},
    "rename": {"ops": [{"op": "rename", "tool": "set_window", "to": "fn_d8abb4"}]},
}

def main():
    for label, spec in SPECS.items():
        comp = compile_spec(TOOLS, spec)
        calls = comp.encode_episode(GOLD)            # how a perfect model acts in this variant
        native, flags = comp.decode_seq(calls)       # what the environment executes
        assert [(n, a) for n, a in native] == [(g["name"], g["arguments"]) for g in GOLD] and not flags
        print(f"== {label}: tools = {[t['name'] for t in comp.tools]}")
        for name, args in calls:
            print(f"   {name}({json.dumps(args)})")

    # The merge/split ladder works on OpenAI-format tools: k=0 coarsest, k=5 native, k=9 finest.
    openai_tools = [{"type": "function", "function": t} for t in TOOLS]
    for k in (0, 5, 9):
        variant_tools, call_map = SchemaAdapter(k=k).transform(openai_tools)
        print(f"ladder k={k}: {[t['function']['name'] for t in variant_tools]}")


if __name__ == "__main__":
    main()
