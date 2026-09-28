"""Whole-catalog class-grouping controls (membership vs naming of class dispatchers).

Four arms, all 12 dispatchers, same sorted member-set size vector, same flat
namespaced merge encoding:

  class_semantic      one dispatcher per native class, named by the class
  class_neutral       same member sets, names group_01..group_12
  random_neutral      size-matched random partitions, neutral names
  anti_class_neutral  size-matched anti-class partitions, neutral names

Operation names stay the native method names. Random and anti partitions are
rejected (and repaired, then reseeded) if a dispatcher would contain two
members with the same method name, so the operation enum stays unique inside
every group and identical in construction across arms.

Native identity is class-qualified (`<class>.<method>`) because five method
names collide across classes.
"""
from __future__ import annotations

import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from .merge_encodings import encode_merged, validate_merged
from .operators import SeqDecoder, compile_spec
from .schema_transform import _execute_tool
from .simenv import SimEnv
TOOLS_PATH = os.path.join(ROOT, "benchmarks", "synthetic", "data", "tools.json")
SPEC_DIR = os.path.join(ROOT, "benchmarks", "synthetic", "data", "specs")
ASSIGN_PATH = os.path.join(SPEC_DIR, "class_grouping_assignments.json")

RANDOM_SEEDS = (101, 102, 103, 104, 105)
ANTI_SEEDS = (201, 202, 203, 204, 205)
N_REALIZATIONS = 5
FILLER = " Neutral dispatch surface."
CLASS_TOKEN_RE = None


def load_catalog(path=TOOLS_PATH):
    rows = json.load(open(path, encoding="utf-8"))
    catalog = []
    for row in rows:
        catalog.append({
            "name": row["name"],
            "description": row.get("description", "") or "",
            "parameters": row["parameters"],
            "group": row["group"],
        })
    return catalog


def class_names(catalog):
    return sorted({row["group"] for row in catalog})


def _by_class(catalog):
    grouped = defaultdict(list)
    for row in catalog:
        grouped[row["group"]].append(row)
    return {name: grouped[name] for name in sorted(grouped)}


def class_sizes(catalog):
    grouped = _by_class(catalog)
    return [len(grouped[name]) for name in sorted(grouped)]


def _has_name_collision(bucket):
    names = [row["name"] for row in bucket]
    return len(names) != len(set(names))


def _repair_name_collisions(buckets, rng):
    """Swap members across same-size-preserving buckets until names are unique."""
    for _ in range(256):
        if not any(_has_name_collision(bucket) for bucket in buckets):
            return True
        progress = False
        for i, bucket in enumerate(buckets):
            counts = Counter(row["name"] for row in bucket)
            for name, count in list(counts.items()):
                if count < 2:
                    continue
                extras = [idx for idx, row in enumerate(bucket) if row["name"] == name][1:]
                for idx in extras:
                    item = bucket[idx]
                    order = list(range(len(buckets)))
                    rng.shuffle(order)
                    swapped = False
                    for j in order:
                        if j == i:
                            continue
                        for other_idx, other in enumerate(buckets[j]):
                            i_names = [row["name"] for k, row in enumerate(bucket) if k != idx]
                            i_names.append(other["name"])
                            j_names = [row["name"] for k, row in enumerate(buckets[j]) if k != other_idx]
                            j_names.append(item["name"])
                            if len(i_names) == len(set(i_names)) and len(j_names) == len(set(j_names)):
                                bucket[idx], buckets[j][other_idx] = other, item
                                swapped = True
                                progress = True
                                break
                        if swapped:
                            break
                    if not swapped:
                        return False
        if not progress:
            return False
    return not any(_has_name_collision(bucket) for bucket in buckets)


def _slice(rows, capacities):
    buckets, offset = [], 0
    for capacity in capacities:
        buckets.append(list(rows[offset:offset + capacity]))
        offset += capacity
    if offset != len(rows):
        raise ValueError(f"capacity sum {offset} != catalog {len(rows)}")
    return buckets


def partition_class(catalog):
    grouped = _by_class(catalog)
    return [list(grouped[name]) for name in sorted(grouped)]


def partition_random(catalog, seed, max_try=200):
    capacities = class_sizes(catalog)
    for delta in range(max_try):
        used = seed + delta
        rng = random.Random(used)
        rows = list(catalog)
        rng.shuffle(rows)
        buckets = _slice(rows, capacities)
        if _repair_name_collisions(buckets, rng):
            return buckets, used
    raise RuntimeError(f"no collision-free random partition from seed {seed}")


def partition_anti(catalog, seed, max_try=200):
    capacities = class_sizes(catalog)
    for delta in range(max_try):
        used = seed + delta
        rng = random.Random(used)
        rows = list(catalog)
        rng.shuffle(rows)
        rows.sort(key=lambda row: row["group"])
        buckets = [[] for _ in capacities]
        domain_counts = [defaultdict(int) for _ in capacities]
        for row in rows:
            candidates = [i for i, cap in enumerate(capacities) if len(buckets[i]) < cap]
            best = min(domain_counts[i][row["group"]] for i in candidates)
            candidates = [i for i in candidates if domain_counts[i][row["group"]] == best]
            max_room = max(capacities[i] - len(buckets[i]) for i in candidates)
            candidates = [i for i in candidates if capacities[i] - len(buckets[i]) == max_room]
            index = rng.choice(candidates)
            buckets[index].append(row)
            domain_counts[index][row["group"]] += 1
        if _repair_name_collisions(buckets, rng):
            return buckets, used
    raise RuntimeError(f"no collision-free anti-class partition from seed {seed}")


def _neutral_name(index):
    return f"group_{index + 1:02d}"


def _group_payload(into, members, description=None):
    payload = {
        "into": into,
        "members": [[row["group"], row["name"]] for row in members],
        "size": len(members),
    }
    if description is not None:
        payload["description"] = description
    return payload


def _semantic_description(members):
    tools = [{"name": row["name"], "description": row["description"],
              "parameters": row["parameters"]} for row in members]
    return _execute_tool(tools, name="execute")["description"]


def _class_token_re(catalog):
    global CLASS_TOKEN_RE
    if CLASS_TOKEN_RE is None:
        names = sorted({row["group"] for row in catalog}, key=len, reverse=True)
        CLASS_TOKEN_RE = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b")
    return CLASS_TOKEN_RE


def _pad_description(description, name, target_chars):
    current = len(name) + len(description)
    if current >= target_chars:
        return description
    need = target_chars - current
    extra = ""
    while len(extra) < need:
        extra += FILLER
    return description + extra[:need]


def _apply_neutral_descriptions(groups, targets):
    """Match name+description character length to the same-size class-semantic group."""
    unused = defaultdict(list)
    for target in targets:
        unused[target["size"]].append(target)
    out = []
    for group in groups:
        pool = unused[group["size"]]
        if not pool:
            raise ValueError(f"no class-semantic target for size {group['size']}")
        target = pool.pop(0)
        description = _pad_description(
            group["description"], group["into"],
            len(target["into"]) + len(target["description"]),
        )
        item = dict(group)
        item["description"] = description
        item["matched_class"] = target["into"]
        out.append(item)
    return out


def build_assignments(catalog=None, random_seeds=RANDOM_SEEDS, anti_seeds=ANTI_SEEDS):
    catalog = list(catalog or load_catalog())
    classes = partition_class(catalog)
    semantic = []
    for index, members in enumerate(classes):
        into = members[0]["group"]
        semantic.append(_group_payload(into, members, _semantic_description(members)))
    neutral = []
    for index, members in enumerate(classes):
        into = _neutral_name(index)
        neutral.append(_group_payload(into, members, _semantic_description(members)))
    neutral = _apply_neutral_descriptions(neutral, semantic)

    random_arms = []
    for real, seed in enumerate(random_seeds):
        buckets, used = partition_random(catalog, seed)
        groups = []
        for index, members in enumerate(buckets):
            groups.append(_group_payload(
                _neutral_name(index), members, _semantic_description(members),
            ))
        random_arms.append({
            "realization": real,
            "seed_requested": seed,
            "seed_used": used,
            "groups": _apply_neutral_descriptions(groups, semantic),
        })

    anti_arms = []
    for real, seed in enumerate(anti_seeds):
        buckets, used = partition_anti(catalog, seed)
        groups = []
        for index, members in enumerate(buckets):
            groups.append(_group_payload(
                _neutral_name(index), members, _semantic_description(members),
            ))
        anti_arms.append({
            "realization": real,
            "seed_requested": seed,
            "seed_used": used,
            "groups": _apply_neutral_descriptions(groups, semantic),
        })

    return {
        "n_native": len(catalog),
        "n_dispatchers": 12,
        "class_sizes": class_sizes(catalog),
        "class_names": class_names(catalog),
        "class_semantic": {"groups": semantic},
        "class_neutral": {"groups": neutral},
        "random_neutral": random_arms,
        "anti_class_neutral": anti_arms,
    }


def spec_from_groups(arm, groups, realization=None, seed_used=None):
    spec = {"arm": arm, "encoding": "flat", "groups": groups}
    if realization is not None:
        spec["realization"] = realization
    if seed_used is not None:
        spec["seed_used"] = seed_used
    return spec


def all_specs(assignments=None):
    assignments = assignments or build_assignments()
    specs = {
        "class_semantic": spec_from_groups("class_semantic", assignments["class_semantic"]["groups"]),
        "class_neutral": spec_from_groups("class_neutral", assignments["class_neutral"]["groups"]),
    }
    for row in assignments["random_neutral"]:
        tag = f"random_neutral_r{row['realization']}"
        specs[tag] = spec_from_groups(
            "random_neutral", row["groups"], row["realization"], row["seed_used"],
        )
    for row in assignments["anti_class_neutral"]:
        tag = f"anti_class_neutral_r{row['realization']}"
        specs[tag] = spec_from_groups(
            "anti_class_neutral", row["groups"], row["realization"], row["seed_used"],
        )
    return specs


def write_specs(assignments=None):
    os.makedirs(SPEC_DIR, exist_ok=True)
    assignments = assignments or build_assignments()
    with open(ASSIGN_PATH, "w", encoding="utf-8") as handle:
        json.dump(assignments, handle, ensure_ascii=False, indent=2)
    written = [ASSIGN_PATH]
    for tag, spec in all_specs(assignments).items():
        path = os.path.join(SPEC_DIR, f"{tag}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle, ensure_ascii=False, indent=2)
        written.append(path)
    return written


def load_spec(path):
    if isinstance(path, dict):
        return path
    return json.load(open(path, encoding="utf-8"))


class CatalogCompiled:
    """Whole-catalog merge of 12 independently compiled flat dispatchers."""

    def __init__(self, tools, call_map, owner, owner_inv, spec):
        self.tools = tools
        self.call_map = call_map
        self.owner = owner
        self.owner_inv = owner_inv
        self.spec = spec

    def encode_episode(self, domain, gold_calls):
        seq = []
        for gold in gold_calls:
            dispatcher = self.owner_inv[(domain, gold["name"])]
            seq.append(encode_merged("flat", dispatcher, gold["name"], gold.get("arguments", {})))
        return seq

    def decode_qualified(self, seq):
        decoder = SeqDecoder(self.call_map)
        native = []
        for name, args in seq:
            event = decoder.step(name, args)
            if event is None:
                continue
            op, payload = event
            cls = self.owner.get(name, {}).get(op)
            native.append((f"{cls}.{op}" if cls else op, payload))
        return native, decoder.flush()

    def size_vector(self):
        return sorted(len(mapping) for mapping in self.owner.values())


def compile_groups(groups, catalog=None):
    catalog = catalog or load_catalog()
    by_key = {(row["group"], row["name"]): row for row in catalog}
    tools, call_map = [], {}
    owner = {}
    owner_inv = {}
    for group in groups:
        members = []
        mapping = {}
        for cls, name in group["members"]:
            row = by_key[(cls, name)]
            members.append({
                "name": row["name"],
                "description": row["description"],
                "parameters": row["parameters"],
            })
            mapping[name] = cls
            owner_inv[(cls, name)] = group["into"]
        spec = {"ops": [{
            "op": "merge",
            "tools": [row["name"] for row in members],
            "into": group["into"],
            "encoding": "flat",
        }]}
        compiled = compile_spec(members, spec)
        for tool in compiled.tools:
            if tool["name"] == group["into"] and group.get("description"):
                tool = dict(tool)
                tool["description"] = group["description"]
            tools.append(tool)
        for name, entry in compiled.call_map.items():
            call_map[name] = dict(entry)
        owner[group["into"]] = mapping
    return CatalogCompiled(tools, call_map, owner, owner_inv, {"groups": groups})


def compile_spec_file(path, catalog=None):
    spec = load_spec(path)
    compiled = compile_groups(spec["groups"], catalog=catalog)
    compiled.spec = spec
    return compiled


class QualifyingEnv:
    """SimEnv wrapper that scores in class-qualified native space."""

    def __init__(self, compiled, control="hard"):
        self.compiled = compiled
        self.env = SimEnv(compiled.call_map, control=control)
        self.native_actions = []
        self.n_calls = 0
        self.n_rejected = 0

    def call(self, name, args):
        before = len(self.env.native_actions)
        result = self.env.call(name, args)
        self.n_calls = self.env.n_calls
        self.n_rejected = self.env.n_rejected
        if len(self.env.native_actions) > before:
            op, payload = self.env.native_actions[-1]
            cls = self.compiled.owner.get(name, {}).get(op)
            qualified = f"{cls}.{op}" if cls else op
            self.native_actions.append((qualified, payload))
        return result

    def finish(self):
        return self.env.finish()


def openai_tools(compiled):
    return [{"type": "function", "function": tool} for tool in compiled.tools]


def contains_class_token(text, catalog):
    return bool(_class_token_re(catalog).search(text or ""))


def hard_reject_ok(compiled):
    env = QualifyingEnv(compiled, control="hard")
    tested = 0
    for name, entry in compiled.call_map.items():
        if entry.get("kind") != "execute":
            continue
        ops = list(entry["ops"])
        if not ops:
            return False, f"{name} has no ops"
        if validate_merged(entry, {"operation": "not_a_real_op"}) is None:
            return False, f"{name} accepted unknown operation"
        before = env.n_rejected
        env.call(name, {"operation": "not_a_real_op"})
        if env.n_rejected <= before:
            return False, f"{name} SimEnv accepted unknown operation"
        tested += 1
    if tested != 12:
        return False, f"expected 12 dispatchers, tested {tested}"
    return True, ""
