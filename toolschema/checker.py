def _eq(pred, acc):           # pred matches one of the acceptable values (light numeric/string normalization)
    for a in acc:
        if isinstance(a, bool) or isinstance(pred, bool):
            if a is pred: return True
        elif isinstance(a, (int, float)) and isinstance(pred, (int, float)):
            if float(a) == float(pred): return True
        elif isinstance(a, str) and isinstance(pred, str):
            if a.strip() == pred.strip(): return True
        elif a == pred:
            return True
    return False

def check_call(pred_name, pred_args, gold_name, gold_args):
    """BFCL-style single-call equivalence. gold_args = {param: [acceptable values...]}.
       '' in the list => the parameter may be omitted."""
    if pred_name != gold_name: return False
    if set(pred_args) - set(gold_args): return False     # no hallucinated parameters
    for p, acc in gold_args.items():
        if p in pred_args:
            if not _eq(pred_args[p], acc): return False
        elif "" not in acc:                               # missing but required
            return False
    return True