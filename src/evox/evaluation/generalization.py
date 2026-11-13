"""Phase D1 analysis: algorithm transfer (RQ6/H5) and dataset portability (RQ7).

Everything here is a pure function over tables produced by Phase B/C runs, so
it can be unit-tested without training a model.  Two rules are enforced:

* The advantage is never averaged away.  Every table keeps per-scenario-kind
  rows next to the ``ALL`` aggregate, and portability is *classified*
  (holds / shrinks / absent / reversed) rather than reduced to one number.
* Synthetic, quick or under-powered runs can never support a hypothesis.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

FIXED_POLICIES = ("always_retrain", "always_incremental", "always_replay", "ssf_inspired", "active_incremental")
F1_MARGIN = 0.01          # smallest F1 difference treated as a real advantage
RETAIN_FRACTION = 0.5     # advantage must keep >= 50% of the reference dataset's size to "hold"
NONINFERIORITY = 0.02     # same margin Phase C uses for H2
EPS = 1e-9                # float tolerance so a difference of exactly the margin is not lost to rounding
MIN_PAIRS = 6             # Phase C's rule: fewer paired seeds cannot reach p<0.05 after correction

REQUIRED_COLUMNS = ("system", "scenario", "seed", "kind", "macro_f1", "updates", "training_seconds")


# ---------------------------------------------------------------------------
# Dataset portability (RQ7)
# ---------------------------------------------------------------------------
def _check_columns(frame: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(f"Phase C per-scenario table lacks columns: {missing}")


def _paired_p(diff: np.ndarray):
    if len(diff) < MIN_PAIRS:
        return None
    if np.allclose(diff, 0):
        return 1.0
    return float(wilcoxon(diff, zero_method="pratt", method="auto").pvalue)


def advantage_table(per_scenario: pd.DataFrame, dataset: str, algorithm: str,
                    system: str = "evox", baselines: Iterable[str] | None = None) -> pd.DataFrame:
    """EVO-X minus each baseline, per drift kind and pooled (``kind == 'ALL'``).

    Differences are formed per scenario, then averaged within seed, so seeds
    (not scenarios) are the unit of replication, as in Phase C.  The extra
    baseline ``best_fixed`` is the strictest comparison: for each scenario it
    takes the best-scoring fixed policy in hindsight.
    """
    _check_columns(per_scenario)
    if system not in set(per_scenario.system):
        raise ValueError(f"System {system!r} not present in the run")
    present = [b for b in (baselines or FIXED_POLICIES) if b in set(per_scenario.system)]
    wide = per_scenario.pivot_table(index=["scenario", "seed", "kind"], columns="system",
                                    values=["macro_f1", "updates", "training_seconds"], aggfunc="mean")
    rows = []
    comparators = list(present) + (["best_fixed"] if present else []) + (["static"] if "static" in set(per_scenario.system) else [])
    for baseline in comparators:
        if baseline == "best_fixed":
            ref_f1 = wide["macro_f1"][present].max(axis=1)
            # Updates / seconds of the policy that happened to win, for cost context.
            winner = wide["macro_f1"][present].idxmax(axis=1)
            ref_upd = pd.Series([wide["updates"].loc[i, w] for i, w in winner.items()], index=wide.index)
            ref_sec = pd.Series([wide["training_seconds"].loc[i, w] for i, w in winner.items()], index=wide.index)
        else:
            ref_f1, ref_upd, ref_sec = (wide[m][baseline] for m in ("macro_f1", "updates", "training_seconds"))
        frame = pd.DataFrame(dict(f1_evox=wide["macro_f1"][system], f1_base=ref_f1,
                                  upd_evox=wide["updates"][system], upd_base=ref_upd,
                                  sec_evox=wide["training_seconds"][system], sec_base=ref_sec)).dropna()
        frame = frame.reset_index()
        for kind, group in [("ALL", frame)] + list(frame.groupby("kind")):
            seed_mean = group.groupby("seed")[["f1_evox", "f1_base", "upd_evox", "upd_base", "sec_evox", "sec_base"]].mean()
            diff = (seed_mean.f1_evox - seed_mean.f1_base).to_numpy()
            adv = float(np.mean(diff)) if len(diff) else float("nan")
            sec_b = float(seed_mean.sec_base.mean()) if len(seed_mean) else float("nan")
            rows.append(dict(
                dataset=dataset, algorithm=algorithm, kind=kind, baseline=baseline,
                n_scenarios=int(group.scenario.nunique()), n_seeds=int(len(seed_mean)),
                f1_evox=float(seed_mean.f1_evox.mean()), f1_base=float(seed_mean.f1_base.mean()),
                advantage=adv, advantage_std=float(np.std(diff, ddof=1)) if len(diff) > 1 else float("nan"),
                p_wilcoxon=_paired_p(diff),
                updates_evox=float(seed_mean.upd_evox.mean()), updates_base=float(seed_mean.upd_base.mean()),
                seconds_evox=float(seed_mean.sec_evox.mean()), seconds_base=sec_b,
                cost_ratio=float(seed_mean.sec_evox.mean() / sec_b) if sec_b and math.isfinite(sec_b) and sec_b > 0 else None,
                noninferior=bool(adv >= -NONINFERIORITY - EPS) if math.isfinite(adv) else None))
    return pd.DataFrame(rows)


def classify_portability(reference: float, observed: float, margin: float = F1_MARGIN,
                         retain: float = RETAIN_FRACTION) -> str:
    """Where does an advantage measured on the reference dataset stand elsewhere?

    ``holds``      same direction, at least ``retain`` of the reference size
    ``shrinks``    same direction but smaller than ``retain`` of the reference
    ``absent``     reference had an advantage; here it is within +-margin
    ``reversed``   reference had an advantage; here the baseline is better by > margin
    ``emerges``    no advantage on the reference, but one here
    ``none``       no advantage on either (not evidence of portability)
    ``unknown``    a value is missing / not finite
    """
    if not (reference is not None and observed is not None and math.isfinite(reference) and math.isfinite(observed)):
        return "unknown"
    had, has = reference >= margin - EPS, observed >= margin - EPS
    if had and has:
        return "holds" if observed >= retain * reference else "shrinks"
    if had:
        return "reversed" if observed <= -margin + EPS else "absent"
    return "emerges" if has else "none"


def portability_table(advantages: pd.DataFrame, reference_dataset: str, margin: float = F1_MARGIN,
                      retain: float = RETAIN_FRACTION) -> pd.DataFrame:
    """Add ``reference_advantage`` and ``portability`` columns to an advantage table."""
    required = {"dataset", "algorithm", "kind", "baseline", "advantage"}
    if not required <= set(advantages.columns):
        raise ValueError("Advantage table lacks required columns")
    if reference_dataset not in set(advantages.dataset):
        raise ValueError(f"Reference dataset {reference_dataset!r} not among {sorted(set(advantages.dataset))}")
    key = ["algorithm", "kind", "baseline"]
    ref = advantages[advantages.dataset == reference_dataset].set_index(key).advantage
    out = advantages.copy()
    out["reference_advantage"] = [ref.get(tuple(r[k] for k in key), float("nan")) for _, r in out.iterrows()]
    out["portability"] = [
        "reference" if r.dataset == reference_dataset else
        classify_portability(r.reference_advantage, r.advantage, margin, retain) for _, r in out.iterrows()]
    return out


# ---------------------------------------------------------------------------
# Algorithm transfer (RQ6 / H5)
# ---------------------------------------------------------------------------
def transfer_ratio(f1_static: float, f1_native: float, f1_transfer: float, min_gap: float = F1_MARGIN):
    """Share of the native controller's gain over the static IDS that the transferred controller keeps.

    ``None`` when the native controller itself barely beats the static model:
    there is no gain to transfer and a ratio would be numerically meaningless.
    """
    gain = f1_native - f1_static
    if not math.isfinite(gain) or gain < min_gap:
        return None
    return float((f1_transfer - f1_static) / gain)


def decision_agreement(native: Mapping[str, str], transferred: Mapping[str, str]) -> dict:
    """Fraction of identical decision points where both controllers chose the same action."""
    shared = sorted(set(native) & set(transferred))
    if not shared:
        return dict(n_decisions=0, agreement=None)
    same = sum(native[k] == transferred[k] for k in shared)
    return dict(n_decisions=len(shared), agreement=same / len(shared))


def utility_rank_correlation(native: Mapping[str, float], transferred: Mapping[str, float]):
    """Spearman correlation of the two controllers' utility scores over the same eligible actions."""
    shared = sorted(set(native) & set(transferred))
    if len(shared) < 3:
        return None
    a, b = np.array([native[k] for k in shared]), np.array([transferred[k] for k in shared])
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None  # a constant ranking carries no information
    rho = spearmanr(a, b).statistic
    return float(rho) if math.isfinite(rho) else None


def summarize_transfer(per_scenario: pd.DataFrame, agreement: pd.DataFrame, data_tag: str, quick: bool,
                       native: str = "evox_native", transferred: str = "evox_transfer",
                       static: str = "static") -> dict:
    """Seed-level transfer summary and a cautious H5 verdict."""
    _check_columns(per_scenario)
    systems = set(per_scenario.system)
    for name in (native, transferred, static):
        if name not in systems:
            raise ValueError(f"System {name!r} missing from the transfer results")
    by_seed = per_scenario.groupby(["system", "seed"]).macro_f1.mean().unstack(0)
    ratios = [transfer_ratio(r[static], r[native], r[transferred]) for _, r in by_seed.iterrows()]
    valid = [x for x in ratios if x is not None]
    pooled = transfer_ratio(by_seed[static].mean(), by_seed[native].mean(), by_seed[transferred].mean())
    gap = (by_seed[transferred] - by_seed[native]).to_numpy()
    p = _paired_p(gap)
    out = dict(
        n_seeds=int(len(by_seed)), data_tag=data_tag, quick=bool(quick),
        macro_f1={s: float(by_seed[s].mean()) for s in (static, native, transferred)},
        transfer_minus_native=float(np.mean(gap)) if len(gap) else None,
        p_transfer_vs_native=p,
        transfer_ratio_pooled=pooled, transfer_ratio_per_seed=[None if x is None else float(x) for x in ratios],
        action_agreement=float(agreement.agreement.mean()) if len(agreement) and agreement.agreement.notna().any() else None,
        n_decision_points=int(agreement.n_decisions.sum()) if len(agreement) else 0,
        utility_rank_correlation=float(agreement.rank_correlation.mean())
        if len(agreement) and agreement.rank_correlation.notna().any() else None,
    )
    if data_tag != "REAL" or quick or len(by_seed) < MIN_PAIRS:
        status, why = "inconclusive", "Synthetic/quick runs or fewer than six paired seeds validate the pipeline only."
    elif pooled is None:
        status, why = "inconclusive", "The native controller does not beat the static IDS, so there is no gain to transfer."
    elif pooled >= 0.8 and out["transfer_minus_native"] >= -NONINFERIORITY - EPS:
        status, why = "supported", "The transferred controller keeps at least 80% of the native controller's gain over static."
    elif pooled < 0.5:
        status, why = "not_supported", "The transferred controller keeps less than half of the native gain."
    else:
        status, why = "inconclusive", "Partial transfer (between 50% and 80% of the native gain)."
    out["H5"] = dict(status=status, evidence=why)
    return out
