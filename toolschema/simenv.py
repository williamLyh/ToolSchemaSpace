"""SimEnv — simulated executable environment for agentic eval over any variant.

One SimEnv per episode. It wraps a SeqDecoder over the variant's call_map and
gives every model tool call an executable semantics + a tool-result message:

  - resolver calls   -> look up the query in the indirection table (exact then
                        fuzzy); return {"handle": ...} or an error + candidates
  - begin/set/commit -> transaction lifecycle; begin returns a txn_id; commit
                        executes the assembled native action
  - plain / execute / pred calls -> execute immediately (native action)
  - hard control: off-schema names, off-enum `operation`s and out-of-interval
                  predicate calls are REJECTED with an error message (the
                  agentic analogue of hard-control scoring — the model may
                  retry); loose control executes them anyway
  - every executed native action gets a deterministic ack so the model can
    proceed; the synthetic env has no deeper state to report

Scoring is state-based at episode end: the multiset of executed native actions
vs the gold calls, plus the decoder's failure flags (dangling_txn etc.).
"""
import difflib
import json

import sys as _sys
import os as _os

from .operators import SeqDecoder, _in_interval


class SimEnv:
    def __init__(self, call_map, control="hard"):
        self.cmap = call_map
        self.control = control
        self.dec = SeqDecoder(call_map)
        self.native_actions = []                   # executed (name, args)
        self.n_calls = 0
        self.n_rejected = 0

    # ------------------------------------------------------------------ #
    def call(self, name, args):
        """Execute one model tool call; returns the tool-result STRING."""
        self.n_calls += 1
        args = dict(args) if isinstance(args, dict) else {}
        e = self.cmap.get(name)
        if e is None:
            if self.control == "hard":
                self.n_rejected += 1
                return json.dumps({"error": f"Unknown function `{name}`. "
                                            f"Use only the provided tools."})
            self.native_actions.append((name, args))
            return json.dumps({"status": "ok"})

        kind = e.get("kind", "execute" if e.get("namespaced") else "named")

        if kind == "resolver":
            return self._resolve(e, args)

        # hard-control schema-domain validation BEFORE side effects
        if self.control == "hard":
            if kind == "execute" and e.get("ops") is not None:
                from .merge_encodings import validate_merged
                err = validate_merged(e, args)
                if err:
                    self.n_rejected += 1
                    return json.dumps({"error": err})
            bad = [(p, lo, hi) for p, lo, hi in e.get("preds", [])
                   if not _in_interval(args.get(p), lo, hi)]
            if bad:
                p, lo, hi = bad[0]
                self.n_rejected += 1
                return json.dumps({"error": f"`{name}` only handles {p} in "
                                            f"[{lo if lo is not None else '-inf'}, "
                                            f"{hi if hi is not None else 'inf'}); "
                                            f"got {args.get(p)!r}. Call the right variant."})

        n_flags = len(self.dec.flags)
        native = self.dec.step(name, args)
        new_flags = self.dec.flags[n_flags:]

        if kind == "curry_begin":
            return json.dumps({"txn_id": self.dec.last_txn_id,
                               "status": f"transaction open; set arguments then commit"})
        if kind == "curry_set":
            if new_flags:
                self.n_rejected += 1
                return json.dumps({"error": new_flags[-1][1],
                                   "hint": "begin a transaction first and use its txn_id"})
            return json.dumps({"status": "set"})
        if kind == "curry_commit":
            if native is None:
                self.n_rejected += 1
                return json.dumps({"error": (new_flags[-1][1] if new_flags
                                             else "commit failed")})
            missing = [f for f in new_flags if f[0] == "commit_missing_required"]
            if missing and self.control == "hard":
                # roll the commit back: required args absent
                t = self.dec.txns.get(args.get("txn_id"))
                if t:
                    t["done"] = False
                self.n_rejected += 1
                return json.dumps({"error": f"cannot commit: missing required "
                                            f"arguments {missing[0][1].split(':', 1)[1]}"})
            self.native_actions.append(native)
            return json.dumps({"status": "committed"})

        # plain / execute / pred (loose-mode violations fall through to here)
        if native is not None and native[0] is not None:
            self.native_actions.append(native)
            return json.dumps({"status": "ok", "executed": native[0]})
        self.n_rejected += 1
        return json.dumps({"error": "call could not be executed"})

    # ------------------------------------------------------------------ #
    def _resolve(self, e, args):
        q = str(args.get("query", "")).strip()
        table = e.get("table", {})                 # handle -> value
        by_val = {str(v).lower(): h for h, v in table.items()}
        if q.lower() in by_val:
            return json.dumps({"handle": by_val[q.lower()]})
        close = difflib.get_close_matches(q.lower(), list(by_val), n=3, cutoff=0.6)
        if len(close) == 1 or (close and difflib.SequenceMatcher(
                None, q.lower(), close[0]).ratio() >= 0.85):
            return json.dumps({"handle": by_val[close[0]],
                               "matched": table[by_val[close[0]]]})
        if close:
            return json.dumps({"error": "ambiguous or no exact match",
                               "candidates": [table[by_val[c]] for c in close]})
        return json.dumps({"error": f"no {e['param']} matching {q!r}"})

    # ------------------------------------------------------------------ #
    def finish(self):
        """End of episode: flush dangling transactions, return flag list."""
        return self.dec.flush()
