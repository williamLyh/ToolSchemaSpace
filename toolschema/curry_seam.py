"""Stateful curry seam for REAL environments (tau2 first).

Activated by SCHEMA_TAU2_CURRY=1. Every >=2-param tool in the env's catalog is
transactionalized (begin/set*/commit, operators.compile_spec) — the transaction
operator ported out of the synthetic env. Meta calls (begin/set) are answered
HERE, inside one agent turn, SimEnv-style; only committed native calls (and
untouched 0/1-param tools) surface to the orchestrator, so the recorded
trajectory, execution and scoring stay 100% native.

Cross-turn transaction state is kept in a per-conversation decoder cache keyed
by the (system, first-user) message pair, so a txn opened in one agent turn
can still be committed in the next.

Control: SCHEMA_CONTROL=hard (default) rejects protocol violations with an
error tool-result the model may recover from; loose answers "ok" but a commit
still only fires when the decoder can assemble the native call.
"""
import hashlib
import json
import os
from collections import OrderedDict

from .operators import compile_spec

INSTRUCTION = ("Some actions are transactional: call the begin_* function first, "
               "read the returned txn_id, set each argument with the set_* "
               "functions (passing that exact txn_id), then call commit_* to "
               "execute. Nothing happens until commit.")

_CACHE = OrderedDict()          # conv key -> SeqDecoder (LRU, cap 256)
_COMPILED = {}                  # catalog-signature -> Compiled


def _canon(tools_schema):
    return [t["function"] if "function" in t else t for t in tools_schema]


def _compiled_for(tools_schema):
    fns = _canon(tools_schema)
    sig = hashlib.sha1(json.dumps(sorted(f["name"] for f in fns)).encode()).hexdigest()
    if sig not in _COMPILED:
        ops = [{"op": "curry", "tool": f["name"]} for f in fns
               if len(f.get("parameters", {}).get("properties", {})) >= 2]
        _COMPILED[sig] = compile_spec(fns, {"ops": ops})
    return _COMPILED[sig]


def _conv_key(messages):
    sys_c = next((m.get("content") or "" for m in messages if m.get("role") == "system"), "")
    usr_c = next((m.get("content") or "" for m in messages if m.get("role") == "user"), "")
    return hashlib.sha1((str(sys_c) + "\x00" + str(usr_c)).encode()).hexdigest()


class CurrySeam:
    MAX_META_ROUNDS = int(os.environ.get("SCHEMA_TAU2_MAXMETA", "8"))

    def __init__(self, comp, dec, control):
        self.comp = comp
        self.cmap = comp.call_map
        self.dec = dec
        self.control = control
        self.tools = [{"type": "function", "function": t} for t in comp.tools]
        self.decoded = []                     # [(id, native_name, native_args)]
        self.n_meta = 0
        self.n_rejected = 0

    @classmethod
    def get(cls, messages, tools_schema):
        comp = _compiled_for(tools_schema)
        key = _conv_key(messages)
        dec = _CACHE.get(key)
        if dec is None or dec.cmap is not comp.call_map:
            dec = comp.decoder()
            _CACHE[key] = dec
            while len(_CACHE) > 256:
                _CACHE.popitem(last=False)
        else:
            _CACHE.move_to_end(key)
        return cls(comp, dec, os.environ.get("SCHEMA_CONTROL", "hard"))

    # ------------------------------------------------------------------ #
    def _answer_meta(self, name, args):
        """SimEnv-style synthetic tool result for one meta/violating call."""
        e = self.cmap.get(name)
        if e is None:
            self.n_rejected += 1
            return json.dumps({"error": f"Unknown function `{name}`. "
                                        f"Use only the provided tools."})
        kind = e.get("kind")
        n_flags = len(self.dec.flags)
        native = self.dec.step(name, dict(args or {}))
        new = self.dec.flags[n_flags:]
        if kind == "curry_begin":
            return json.dumps({"txn_id": self.dec.last_txn_id,
                               "status": "transaction open; set arguments then commit"})
        if kind == "curry_set":
            if new and self.control == "hard":
                self.n_rejected += 1
                return json.dumps({"error": new[-1][1],
                                   "hint": "begin a transaction first and use its txn_id"})
            return json.dumps({"status": "set"})
        if kind == "curry_commit":                     # failed commit lands here
            if native is None:
                self.n_rejected += 1
                return json.dumps({"error": (new[-1][1] if new else "commit failed")})
            missing = [f for f in new if f[0] == "commit_missing_required"]
            if missing and self.control == "hard":
                t = self.dec.txns.get((args or {}).get("txn_id"))
                if t:
                    t["done"] = False
                self.n_rejected += 1
                return json.dumps({"error": f"cannot commit: missing required arguments "
                                            f"{missing[0][1].split(':', 1)[1]}"})
            return None, native                       # sentinel: commit succeeded
        return json.dumps({"status": "ok"})

    def process(self, tool_calls):
        """Split one model response's calls into (meta_answers, native_calls).

        meta_answers: [(id, name, args_json, result_str)] to be answered locally.
        native_calls: [(id, native_name, native_args)] to surface.
        """
        metas, natives = [], []
        for tc in tool_calls:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            e = self.cmap.get(name)
            kind = e.get("kind") if e else None
            if kind == "curry_commit":
                r = self._answer_meta(name, args)
                if isinstance(r, tuple):              # success -> native action
                    natives.append((tc.id, r[1][0], r[1][1]))
                else:
                    metas.append((tc.id, name, tc.function.arguments, r))
                continue
            if kind in ("curry_begin", "curry_set") or e is None:
                if e is None and self.control != "hard":
                    natives.append((tc.id, name, args))   # loose passthrough
                else:
                    self.n_meta += 1
                    metas.append((tc.id, name, tc.function.arguments,
                                  self._answer_meta(name, args)))
                continue
            # plain named tool (0/1-param, untouched): decode via SeqDecoder
            native = self.dec.step(name, args)
            if native is not None:
                natives.append((tc.id, native[0], native[1]))
        return metas, natives
