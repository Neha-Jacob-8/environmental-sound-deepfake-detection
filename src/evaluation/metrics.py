"""Detection metrics for EnvSDD.

EER is the primary metric for the ESDD protocol: the operating point where the
false-acceptance and false-rejection rates coincide. It is threshold-free, which
matters here because a threshold tuned on seen generators does not transfer to
unseen ones - the whole question under study.

Convention throughout: label 1 = fake, label 0 = real, and `score` is the model's
fake-ness score (higher = more likely fake).
"""

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def eer(y_true, y_score):
    """Equal error rate and the threshold that achieves it.

    Interpolates between the two ROC points that bracket fpr == fnr rather than
    snapping to the nearest one, so the estimate does not jump in coarse steps
    when the score set is small.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")

    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    diff = fnr - fpr

    # diff runs from +1 down to -1, so a sign change is guaranteed to exist.
    i = np.nanargmin(np.abs(diff))
    if diff[i] == 0:
        return float((fpr[i] + fnr[i]) / 2), float(thr[i])

    # Pick the neighbour on the other side of zero and interpolate linearly.
    j = i + 1 if (i + 1 < len(diff) and diff[i] * diff[i + 1] < 0) else i - 1
    if j < 0 or j >= len(diff) or diff[i] * diff[j] > 0:
        return float((fpr[i] + fnr[i]) / 2), float(thr[i])

    t = diff[i] / (diff[i] - diff[j])          # in [0, 1]
    rate = (fpr[i] + t * (fpr[j] - fpr[i]) + fnr[i] + t * (fnr[j] - fnr[i])) / 2
    return float(rate), float(thr[i] + t * (thr[j] - thr[i]))


def all_metrics(y_true, y_score, threshold=None):
    """EER, AUC and F1. F1 needs a threshold; defaults to the EER threshold.

    F1 at the EER threshold is reported for comparability with the literature,
    but note it is computed at an operating point chosen on this same data, so
    it flatters the model slightly. EER and AUC are the threshold-free numbers.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    e, thr = eer(y_true, y_score)
    if threshold is None:
        threshold = thr
    y_pred = (y_score >= threshold).astype(int)
    return {
        "eer": e,
        "auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "n_real": int((y_true == 0).sum()),
        "n_fake": int((y_true == 1).sum()),
    }


def per_generator_eer(y_true, y_score, generators):
    """EER for each generator, scored against the *same* pool of real clips.

    Every fake generator is evaluated one at a time against all real clips, so
    the numbers are comparable across generators: only the fake side changes.
    Pooling all fakes into a single EER would let an easy generator mask a hard
    one, which is exactly the failure this project is trying to expose.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    generators = np.asarray(generators)

    real = y_true == 0
    out = {}
    for g in sorted(set(generators[y_true == 1])):
        sel = real | (generators == g)
        out[g] = all_metrics(y_true[sel], y_score[sel])
    return out


def seen_unseen_summary(y_true, y_score, generators, seen, unseen):
    """Pooled EER over seen vs unseen generators, plus the gap between them.

    The gap is the headline number: how much worse the detector gets on
    generators it never trained on.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    generators = np.asarray(generators)
    real = y_true == 0

    out = {}
    for name, group in (("seen", seen), ("unseen", unseen)):
        sel = real | np.isin(generators, list(group))
        out[name] = all_metrics(y_true[sel], y_score[sel])
    out["gap"] = out["unseen"]["eer"] - out["seen"]["eer"]
    return out
