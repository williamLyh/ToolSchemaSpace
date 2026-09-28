"""Typed single-call encodings of a merged member set.

The existing merge operator always emits a flat namespaced dispatcher:

    execute(operation, set_temp::room, set_temp::value, ...)

This module adds two confirmatory encodings of the *same* member set:

    nested  execute(operation, arguments={room, value, ...})
    union   execute(set_temp={...}) xor execute(set_fan={...})

Neither encoding uses free-form JSON strings or code. Nested does not depend
on JSON-Schema oneOf: `arguments` is a typed object whose properties are the
union of member parameters, and hard validation rejects keys that do not
belong to the selected operation.
"""
from .schema_transform import _gprops, _greq


ENCODINGS = ("flat", "nested", "union")


def _present_obj(value):
    """A union-encoding operation key is present only if it is a non-empty object."""
    return isinstance(value, dict) and len(value) > 0


def execute_tool_nested(merged, name="execute"):
    """operation enum + one nested object of native (non-namespaced) arguments."""
    union_props = {}
    for tool in merged:
        for pname, pdef in _gprops(tool).items():
            if pname in union_props:
                continue
            copied = dict(pdef)
            copied["description"] = (
                f"[{tool['name']}] " + copied.get("description", "")
            )
            union_props[pname] = copied
    ops = ", ".join(tool["name"] for tool in merged)
    return {
        "name": name,
        "description": (
            f"Dispatch entry point handling ONLY these {len(merged)} operations: "
            f"{ops}. Set `operation`, then put that operation's own arguments "
            f"inside `arguments`. Do not namespace argument names and do not "
            f"include arguments for any other operation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": [tool["name"] for tool in merged],
                    "description": "Which operation to perform.",
                },
                "arguments": {
                    "type": "object",
                    "description": (
                        "Arguments for the selected operation only, using that "
                        "operation's original parameter names."
                    ),
                    "properties": union_props,
                    "additionalProperties": False,
                },
            },
            "required": ["operation", "arguments"],
        },
    }


def execute_tool_union(merged, name="execute"):
    """One optional object argument per member operation; exactly one must be set."""
    props = {}
    for tool in merged:
        props[tool["name"]] = {
            "type": "object",
            "description": (
                f"Arguments for `{tool['name']}` only. Provide this object "
                f"for this operation and omit every other operation object. "
                f"{tool.get('description', '')}"
            ).strip(),
            "properties": {k: dict(v) for k, v in _gprops(tool).items()},
            "required": list(_greq(tool)),
            "additionalProperties": False,
        }
    ops = ", ".join(tool["name"] for tool in merged)
    return {
        "name": name,
        "description": (
            f"Dispatch entry point handling ONLY these {len(merged)} operations: "
            f"{ops}. Provide exactly one operation object and leave the others unset."
        ),
        "parameters": {
            "type": "object",
            "properties": props,
            "required": [],
        },
    }


def member_param_index(merged):
    """operation -> sorted native parameter names."""
    return {tool["name"]: list(_gprops(tool)) for tool in merged}


def encode_merged(encoding, entry, native_name, native_args):
    args = dict(native_args)
    if encoding == "flat":
        payload = {"operation": native_name}
        payload.update({f"{native_name}::{key}": value for key, value in args.items()})
        return entry, payload
    if encoding == "nested":
        return entry, {"operation": native_name, "arguments": args}
    if encoding == "union":
        return entry, {native_name: args}
    raise ValueError(f"unknown merge encoding {encoding!r}")


def decode_merged(entry, args):
    """Return (native_name, native_args) or (None, {}) if the call is unusable."""
    encoding = entry.get("encoding", "flat")
    args = dict(args or {})
    if encoding == "nested":
        payload = args.get("arguments")
        if not isinstance(payload, dict):
            payload = {}
        return args.get("operation"), dict(payload)
    if encoding == "union":
        present = [key for key, value in args.items() if _present_obj(value)]
        if len(present) != 1:
            return None, {}
        return present[0], dict(args[present[0]])
    native_name = args.get("operation")
    native_args = {}
    for key, value in args.items():
        if key == "operation" or value in ("", None):
            continue
        native_args[key.split("::", 1)[1] if "::" in key else key] = value
    return native_name, native_args


def validate_merged(entry, args):
    """Hard-validation error string, or None if the call is on-schema."""
    encoding = entry.get("encoding", "flat")
    ops = entry.get("ops")
    params = entry.get("op_params") or {}
    args = dict(args or {})
    if encoding == "union":
        present = [key for key, value in args.items() if _present_obj(value)]
        extra = [key for key in args
                 if key not in (ops or []) and args[key] not in ("", None, {})]
        if extra:
            return f"unknown union keys {extra}"
        if len(present) == 0:
            return "expected exactly one operation object; got none"
        if len(present) > 1:
            return f"expected exactly one operation object; got {present}"
        op = present[0]
        if ops is not None and op not in ops:
            return f"operation must be one of {ops}"
        allowed = set(params.get(op, []))
        unknown = [key for key in args[op] if key not in allowed]
        if unknown:
            return f"arguments {unknown} do not belong to `{op}`"
        return None
    op = args.get("operation")
    if ops is not None and op not in ops:
        return f"operation must be one of {ops}"
    if encoding == "nested":
        payload = args.get("arguments")
        if not isinstance(payload, dict):
            return "`arguments` must be an object"
        allowed = set(params.get(op, []))
        unknown = [key for key in payload if key not in allowed]
        if unknown:
            return f"arguments {unknown} do not belong to `{op}`"
        extra = [key for key in args if key not in ("operation", "arguments")]
        if extra:
            return f"unexpected top-level keys {extra}"
    return None


def schema_complexity(tools):
    """Covariates that differ across encodings of the same member set."""
    arg_counts = [len(_gprops(tool)) for tool in tools]
    optional = 0
    required = 0
    max_depth = 1

    def walk(node, depth):
        nonlocal optional, required, max_depth
        max_depth = max(max_depth, depth)
        props = node.get("properties") or {}
        req = set(node.get("required") or [])
        for name, spec in props.items():
            if name in req:
                required += 1
            else:
                optional += 1
            if isinstance(spec, dict) and spec.get("type") == "object":
                walk(spec, depth + 1)

    for tool in tools:
        walk(tool.get("parameters") or {}, 1)
    rendered = _stable_json(tools)
    return {
        "n_tools": len(tools),
        "n_args": sum(arg_counts),
        "args_mean": (sum(arg_counts) / len(arg_counts)) if arg_counts else 0.0,
        "n_required": required,
        "n_optional": optional,
        "optional_frac": optional / max(1, optional + required),
        "max_nesting": max_depth,
        "schema_chars": len(rendered),
        "schema_tokens_est": max(1, (len(rendered) + 3) // 4),
    }


def _stable_json(value):
    import json
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
