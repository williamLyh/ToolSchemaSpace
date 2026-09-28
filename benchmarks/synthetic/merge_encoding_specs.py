"""Write the merge-encoding specs (argument structure of a merged tool).

  merge_full_{flat,nested,union}   one dispatcher per domain
  merge_sem_{flat,nested,union}    the same semantic ~4-op partitions as mgroup_sem
  mix_{r03,r05}_{flat,nested,union,rot0,rot1,rot2}
        matched convention-mixture test: at each target function-count ratio (0.3x, 0.5x)
        one semantic partition per domain is frozen. Homogeneous arms use one encoding for
        every group; the three mixed rotations give each group each encoding once.

Run: python -m benchmarks.synthetic.merge_encoding_specs
"""
import json
import os
from collections import defaultdict

from .opspecs2 import DOMAINS, _agglomerate, _cluster_name, specs2_for

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "specs")
ASSIGN_PATH = os.path.join(OUT_DIR, "mixture_assignments.json")
ENCODINGS = ("flat", "nested", "union")
RATIOS = (("r03", 0.3), ("r05", 0.5))


def _stamp(ops, domain, encoding):
    out = []
    for op in ops:
        item = dict(op)
        item["group"] = domain
        if item.get("op") == "merge":
            item["encoding"] = encoding
        out.append(item)
    return out


def write_merge_encodings():
    full = defaultdict(list)
    sem = defaultdict(list)
    for domain in DOMAINS:
        name = domain["name"]
        members = [tool["name"] for tool in domain["tools"]]
        for encoding in ("flat", "nested", "union"):
            full[encoding].append({
                "op": "merge",
                "tools": members,
                "into": "execute",
                "encoding": encoding,
                "group": name,
            })
        sem_ops = specs2_for(domain)["mgroup_sem"]["ops"]
        for encoding in ("flat", "nested", "union"):
            sem[encoding].extend(_stamp(sem_ops, name, encoding))

    written = []
    for encoding in ("flat", "nested", "union"):
        for tag, ops in (("merge_full", full[encoding]), ("merge_sem", sem[encoding])):
            path = os.path.join(OUT_DIR, f"{tag}_{encoding}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"ops": ops}, handle, ensure_ascii=False)
            written.append((path, len(ops)))
            print(f"{path}: {len(ops)} ops")
    return written



def target_k(n, ratio):
    return max(2, min(n - 1, round(n * ratio)))


def _partition(domain, ratio):
    tools = list(domain["tools"])
    k = target_k(len(tools), ratio)
    clusters = _agglomerate(tools, k)
    used = set()
    groups = []
    for members in clusters:
        groups.append({
            "into": _cluster_name(members, used),
            "tools": [tool["name"] for tool in members],
        })
    groups.sort(key=lambda item: item["into"])
    return groups


def _merge_op(group, domain, encoding):
    return {
        "op": "merge",
        "tools": list(group["tools"]),
        "into": group["into"],
        "encoding": encoding,
        "group": domain,
    }


def write_encoding_mixtures():
    assignments = {}
    written = []
    for tag, ratio in RATIOS:
        by_domain = {}
        n_native = 0
        n_exposed = 0
        for domain in DOMAINS:
            groups = _partition(domain, ratio)
            by_domain[domain["name"]] = {
                "n_native": len(domain["tools"]),
                "n_exposed": len(groups),
                "groups": groups,
            }
            n_native += len(domain["tools"])
            n_exposed += len(groups)
        assignments[tag] = {
            "ratio_target": ratio,
            "ratio_unweighted": n_exposed / n_native,
            "n_native": n_native,
            "n_exposed": n_exposed,
            "domains": by_domain,
        }

        homog = {encoding: [] for encoding in ENCODINGS}
        mixed = {rot: [] for rot in range(3)}
        for domain, info in by_domain.items():
            for encoding in ENCODINGS:
                for group in info["groups"]:
                    homog[encoding].append(_merge_op(group, domain, encoding))
            for rot in range(3):
                for index, group in enumerate(info["groups"]):
                    encoding = ENCODINGS[(index + rot) % 3]
                    mixed[rot].append(_merge_op(group, domain, encoding))

        for encoding in ENCODINGS:
            path = os.path.join(OUT_DIR, f"mix_{tag}_{encoding}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"ops": homog[encoding]}, handle, ensure_ascii=False)
            written.append((path, len(homog[encoding])))
            print(f"{path}: {len(homog[encoding])} ops")
        for rot in range(3):
            path = os.path.join(OUT_DIR, f"mix_{tag}_rot{rot}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({"ops": mixed[rot]}, handle, ensure_ascii=False)
            written.append((path, len(mixed[rot])))
            print(f"{path}: {len(mixed[rot])} ops  ratio={n_exposed / n_native:.3f}")

    with open(ASSIGN_PATH, "w", encoding="utf-8") as handle:
        json.dump(assignments, handle, ensure_ascii=False, indent=2)
    print(f"assignments: {ASSIGN_PATH}")
    return written



if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    write_merge_encodings()
    write_encoding_mixtures()
