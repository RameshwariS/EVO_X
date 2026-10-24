"""
EVO-X Phase 5 -- synthetic concept-drift generator (Synopsis §7.3 Module 2, §7.9 Phase 5)
=========================================================================================
Why this exists: the natural CICIDS2017 stream is almost entirely *novel-class* drift (attacks are
weekday-scheduled).  Recurring / gradual / incremental / temporary drift of KNOWN classes must be
constructed, and constructed with ground truth.

What each scenario IS (be explicit -- §9.3 limitation):
  * a class-mix change                         = prior / label-distribution shift  (re-sequencing)
  * a bounded additive feature offset          = synthetic covariate shift on informative features
  * their combination in the scenarios below   = *synthetic* concept drift.  It is NOT naturally
    occurring drift and must never be described as such in the report.
  * novel_class                                = a class absent from the reference appears.

Two stages
  LabSource      class-indexed pool of real flows, split (per class) into  train / val / pool_validation /
                 pool_test.  The two pools are DISJOINT so validation scenarios (tune thresholds/weights)
                 and test scenarios (report) never share a single flow.
  generate_scenario(source, spec) -> Scenario
                 an EvoStream-compatible stream + `drift_truth` dict (written as drift_truth.json).

Sampling model.  Two concepts, A (reference-like) and B (shifted).  A schedule p(t) in [0,1] gives, per flow,
the probability of drawing from B.  gradual / recurring / temporary / abrupt / novel_class sample a concept
per flow (a probabilistic blend); incremental additionally ramps the feature offset continuously, giving
small cumulative shifts.  Class priors of concept A equal the TRAINING prior, so nothing is drifting before
the change point (ground truth is clean).
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from evox.data.stream import EvoStream

KINDS = ("abrupt", "gradual", "recurring", "temporary", "incremental", "novel_class")
MAGNITUDES = ("low", "med", "high")
CONTROL_KIND = "stationary"        # not one of the six drift types: a drift-free control used to measure false alarms
# lam   = how far the class mix moves from A towards the attack-heavy target (0..1)
# sigma = additive offset in reference standard deviations on ~10% informative features
# novel_share = share of the new class in the stream after it appears
MAGNITUDE_TABLE = {
    "low":  dict(lam=0.30, sigma=0.75, novel_share=0.03),
    "med":  dict(lam=0.60, sigma=1.50, novel_share=0.08),
    "high": dict(lam=1.00, sigma=3.00, novel_share=0.20),
}
SHIFT_FAMILY = {
    "abrupt": "prior shift + covariate shift (synthetic), instantaneous",
    "gradual": "prior shift + covariate shift (synthetic), probabilistic blend",
    "recurring": "prior shift + covariate shift (synthetic), concept B returns after reverting to A",
    "temporary": "prior shift + covariate shift (synthetic), short-lived then reverts",
    "incremental": "covariate offset ramped continuously (cumulative) + prior shift (synthetic)",
    "novel_class": "novel attack class appears (class absent from the reference)",
    "stationary": "control: no drift at all (concept A throughout); used for false-alarm rates",
}


# ---------------------------------------------------------------------------
# Source pool
# ---------------------------------------------------------------------------
@dataclass
class LabSource:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    pools: Dict[str, Dict[int, np.ndarray]]          # family -> class id -> flows (disjoint between families)
    classes: List[str]
    feature_names: List[str]
    informative_order: np.ndarray                    # feature indices, most class-discriminative first
    dropped_classes: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def class_ids(self) -> List[int]:
        return sorted(np.unique(self.y_train).tolist())

    def pool_classes(self, family: str) -> List[int]:
        return sorted(self.pools[family])

    @classmethod
    def from_evostream(cls, s: EvoStream, seed: int = 0, train_frac: float = 0.4, val_frac: float = 0.1,
                       min_flows: int = 120, within_class: str = "random") -> "LabSource":
        """Pool every flow of an EvoStream (reference + live) and split PER CLASS.

        within_class="random"        clean ground truth: pre-drift data is i.i.d. (default)
        within_class="chronological" keeps natural within-class drift (use for the 'natural' scenario only)
        Features are re-standardised with statistics of the lab TRAIN split only (no pool/val statistics).
        """
        if within_class not in ("random", "chronological"):
            raise ValueError(within_class)
        X = np.vstack([s.X_ref_train, s.X_ref_val, s.X_stream])
        y = np.concatenate([s.y_ref_train, s.y_ref_val, s.y_stream]).astype(np.int64)
        rng = np.random.default_rng([seed, 17])
        tr, va, pv, pt, dropped = [], [], [], [], []
        pool_idx: Dict[str, Dict[int, np.ndarray]] = {"validation": {}, "test": {}}
        for c in np.unique(y):
            idx = np.flatnonzero(y == c)                   # chronological order inside the class
            if len(idx) < min_flows:
                dropped.append(s.classes[c]); continue
            if within_class == "random":
                idx = rng.permutation(idx)
            n_tr, n_va = int(len(idx) * train_frac), max(1, int(len(idx) * val_frac))
            rest = idx[n_tr + n_va:]
            half = len(rest) // 2
            tr.append(idx[:n_tr]); va.append(idx[n_tr:n_tr + n_va])
            pool_idx["validation"][int(c)], pool_idx["test"][int(c)] = rest[:half], rest[half:]
        tr, va = np.concatenate(tr), np.concatenate(va)
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0)
        sd = np.where(sd == 0, 1.0, sd)
        Z = ((X - mu) / sd).astype(np.float32)
        # informativeness = variance of class means (on train only)
        cl = np.unique(y[tr])
        means = np.vstack([Z[tr][y[tr] == c].mean(axis=0) for c in cl])
        order = np.argsort(-means.var(axis=0))
        pools = {f: {c: Z[i] for c, i in d.items()} for f, d in pool_idx.items()}
        return cls(Z[tr], y[tr].astype(np.int16), Z[va], y[va].astype(np.int16), pools, list(s.classes),
                   list(s.feature_names), order, dropped,
                   meta=dict(seed=seed, within_class=within_class, train_frac=train_frac, min_flows=min_flows,
                             n_train=len(tr), n_val=len(va),
                             pool_sizes={f: {s.classes[c]: len(v) for c, v in d.items()} for f, d in pools.items()}))


# ---------------------------------------------------------------------------
# Specification
# ---------------------------------------------------------------------------
@dataclass
class ScenarioSpec:
    kind: str
    magnitude: str = "med"
    seed: int = 0
    family: str = "test"               # 'validation' | 'test'  (selects the disjoint pool half)
    n_batches: int = 120
    batch_size: int = 500
    change_batch: int = 40
    gradual_width: int = 30            # batches over which B takes over (gradual)
    incremental_width: int = 60        # batches over which the offset ramps to full size
    temporary_len: int = 6             # batches B lasts before reverting
    recurring_len: int = 25            # length of each A / B segment after the first change

    def validate(self):
        if self.kind not in KINDS + (CONTROL_KIND,): raise ValueError(f"kind must be one of {KINDS} or '{CONTROL_KIND}'")
        if self.magnitude not in MAGNITUDES: raise ValueError(f"magnitude must be one of {MAGNITUDES}")
        if self.family not in ("validation", "test"): raise ValueError("family")
        c, N = self.change_batch, self.n_batches
        need = {"abrupt": c + 1, "novel_class": c + 1, "gradual": c + self.gradual_width,
                "incremental": c + self.incremental_width, "temporary": c + self.temporary_len + 10,
                "recurring": c + 2 * self.recurring_len + 5, CONTROL_KIND: 0}[self.kind]
        if need >= N:
            raise ValueError(f"{self.kind}: n_batches={N} too short (needs > {need})")

    @property
    def name(self) -> str:
        return f"{self.kind}_{self.magnitude}_{self.family[:3]}{self.seed}"


@dataclass
class Scenario:
    spec: ScenarioSpec
    stream: EvoStream
    truth: Dict
    p_b: np.ndarray                    # per flow: probability of drawing from concept B
    source_ids: Optional[np.ndarray] = None  # source-pool identity, before offsets/resampling (Phase B holdouts)

    def save(self, out_dir: Path, with_stream: bool = True) -> Path:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "drift_truth.json").write_text(json.dumps(self.truth, indent=1))
        if with_stream and self.source_ids is not None:
            np.save(out_dir / "source_ids.npy", self.source_ids, allow_pickle=False)
        if with_stream:
            self.stream.save(out_dir)
        return out_dir


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def _timeline(spec: ScenarioSpec, n_flows: int):
    bs, c = spec.batch_size, spec.change_batch
    t = np.arange(n_flows) / bs                                  # fractional batch index
    k = spec.kind
    if k == CONTROL_KIND:
        return (np.zeros(n_flows), [], [dict(concept="A", start_batch=0, end_batch=spec.n_batches - 1)])
    if k in ("abrupt", "novel_class"):
        p = (t >= c).astype(float)
        ev = [dict(type="onset", start_batch=c, end_batch=c, scored=True, to_concept="B")]
        seg = [("A", 0, c - 1), ("B", c, spec.n_batches - 1)]
    elif k in ("gradual", "incremental"):
        w = spec.gradual_width if k == "gradual" else spec.incremental_width
        p = np.clip((t - c) / w, 0.0, 1.0)
        ev = [dict(type="onset", start_batch=c, end_batch=c + w, scored=True, to_concept="B")]
        seg = [("A", 0, c - 1), ("A->B", c, c + w - 1), ("B", c + w, spec.n_batches - 1)]
    elif k == "temporary":
        s_ = spec.temporary_len
        p = ((t >= c) & (t < c + s_)).astype(float)
        ev = [dict(type="onset", start_batch=c, end_batch=c, scored=True, to_concept="B"),
              dict(type="revert", start_batch=c + s_, end_batch=c + s_, scored=False, to_concept="A")]
        seg = [("A", 0, c - 1), ("B", c, c + s_ - 1), ("A", c + s_, spec.n_batches - 1)]
    else:  # recurring: A, B, A, B
        L = spec.recurring_len
        p = (((t >= c) & (t < c + L)) | (t >= c + 2 * L)).astype(float)
        ev = [dict(type="onset", start_batch=c, end_batch=c, scored=True, to_concept="B"),
              dict(type="revert", start_batch=c + L, end_batch=c + L, scored=False, to_concept="A"),
              dict(type="onset", start_batch=c + 2 * L, end_batch=c + 2 * L, scored=True, to_concept="B", recurrence=True)]
        seg = [("A", 0, c - 1), ("B", c, c + L - 1), ("A", c + L, c + 2 * L - 1), ("B", c + 2 * L, spec.n_batches - 1)]
    for i, e in enumerate(ev):
        e["id"] = i
    return p, ev, [dict(concept=a, start_batch=b, end_batch=e) for a, b, e in seg]


def _mix(a: Dict[int, float], b: Dict[int, float], lam: float) -> Dict[int, float]:
    keys = set(a) | set(b)
    return {k: (1 - lam) * a.get(k, 0.0) + lam * b.get(k, 0.0) for k in keys}


def _tv(a, b) -> float:
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def generate_scenario(source: LabSource, spec: ScenarioSpec) -> Scenario:
    spec.validate()
    mag = MAGNITUDE_TABLE[spec.magnitude]
    rng = np.random.default_rng(np.random.SeedSequence(
        [spec.seed, zlib.crc32(spec.kind.encode()), MAGNITUDES.index(spec.magnitude), spec.family == "test"]))
    # concept choice (which classes / features / novel class) must NOT depend on the magnitude level, otherwise
    # low/med/high would differ in *which* drift they are, not just how strong it is (calibration would be noisy)
    rng_c = np.random.default_rng(np.random.SeedSequence(
        [spec.seed, zlib.crc32(spec.kind.encode()), 7, spec.family == "test"]))
    pool = source.pools[spec.family]
    n_flows = spec.n_batches * spec.batch_size
    d = source.X_train.shape[1]

    # ---- known / novel classes -------------------------------------------------------------
    all_ids = source.pool_classes(spec.family)
    novel: List[int] = []
    if spec.kind == "novel_class":
        cands = [c for c in all_ids if c != 0 and len(pool[c]) >= 200] or [c for c in all_ids if c != 0]
        if len(all_ids) < 3 or not cands:
            raise ValueError("novel_class needs >=3 classes in the pool (benign + 1 known attack + 1 novel)")
        novel = [int(rng_c.choice(cands))]
    known = [c for c in source.class_ids if c not in novel]
    if len(known) < 2:
        raise ValueError("need at least 2 known classes")
    tr_keep = np.isin(source.y_train, known)
    va_keep = np.isin(source.y_val, known)

    # ---- concepts ------------------------------------------------------------------------------
    counts = {c: float((source.y_train == c).sum()) for c in known}
    tot = sum(counts.values())
    A = {c: counts[c] / tot for c in known}                       # == reference prior
    attacks = [c for c in known if c != 0]
    if spec.kind == "novel_class":
        B = _mix(A, {novel[0]: 1.0}, mag["novel_share"])
        sigma, off_feats = 0.0, np.array([], dtype=int)
    else:
        sb = rng_c.choice(attacks, size=max(1, round(len(attacks) / 2)), replace=False) if attacks else np.array([], int)
        target: Dict[int, float] = {}
        benign_share = 0.4 if (0 in known and len(sb)) else (1.0 if not len(sb) else 0.0)
        if 0 in known: target[0] = benign_share
        for c in sb: target[int(c)] = (1.0 - benign_share) / len(sb)
        B = _mix(A, target, mag["lam"])
        sigma = mag["sigma"]
        m = max(3, int(round(0.1 * d)))
        top = source.informative_order[: min(d, 3 * m)]
        off_feats = np.sort(rng_c.choice(top, size=m, replace=False))
    offset = np.zeros(d, dtype=np.float32)
    if len(off_feats):
        offset[off_feats] = (sigma * rng_c.choice([-1.0, 1.0], size=len(off_feats))).astype(np.float32)

    # ---- sampling ------------------------------------------------------------------------------
    p, events, segments = _timeline(spec, n_flows)
    from_b = rng.random(n_flows) < p                               # per-flow concept draw
    y = np.empty(n_flows, dtype=np.int16)
    for conc, sel in ((A, ~from_b), (B, from_b)):
        n = int(sel.sum())
        if n:
            ids = np.array(sorted(conc)); pr = np.array([conc[i] for i in ids]); pr = pr / pr.sum()
            y[sel] = rng.choice(ids, size=n, p=pr)
    X = np.empty((n_flows, d), dtype=np.float32)
    source_ids = np.empty(n_flows, dtype="U64")
    for c in np.unique(y):
        sel = np.flatnonzero(y == c)
        draws = rng.integers(0, len(pool[int(c)]), size=len(sel))
        X[sel] = pool[int(c)][draws]
        source_ids[sel] = [f"{spec.family}:{int(c)}:{int(i)}" for i in draws]
    if spec.kind == "incremental":
        X += p[:, None].astype(np.float32) * offset                 # continuous ramp: cumulative shift
    else:
        X[from_b] += offset                                          # regime offset follows the drawn concept

    # ---- stream object + truth ------------------------------------------------------------------
    batch_p = p.reshape(spec.n_batches, spec.batch_size).mean(axis=1)
    stream = EvoStream(
        X_ref_train=source.X_train[tr_keep], y_ref_train=source.y_train[tr_keep],
        X_ref_val=source.X_val[va_keep], y_ref_val=source.y_val[va_keep],
        X_stream=X, y_stream=y, feature_names=list(source.feature_names), classes=list(source.classes),
        # lab features are already standardised (lab-train statistics) -> identity scaler, so save/load works
        scaler_mean=np.zeros(d), scaler_scale=np.ones(d), impute_median=np.zeros(d),
        meta=dict(dataset="LAB:" + spec.name, ordering_source="synthetic_scenario", batch_size=spec.batch_size,
                  scenario=spec.name, n_stream=n_flows))
    affected = sorted({c for c in set(A) | set(B) if abs(A.get(c, 0) - B.get(c, 0)) > 1e-3})
    truth = dict(
        name=spec.name, kind=spec.kind, magnitude=spec.magnitude, seed=spec.seed, family=spec.family,
        shift_family=SHIFT_FAMILY[spec.kind], n_batches=spec.n_batches, batch_size=spec.batch_size, n_flows=n_flows,
        change_batch=spec.change_batch, known_classes=[source.classes[c] for c in known],
        known_class_ids=[int(c) for c in known], novel_classes=[source.classes[c] for c in novel],
        novel_class_ids=[int(c) for c in novel],
        first_batch_of_novel_class=spec.change_batch if novel else None,
        concepts=dict(A=dict(mix={source.classes[c]: round(v, 6) for c, v in A.items()}, offset_sigma=0.0),
                      B=dict(mix={source.classes[c]: round(v, 6) for c, v in B.items() if v > 0}, offset_sigma=float(sigma),
                             offset_features=[source.feature_names[i] for i in off_feats])),
        magnitude_detail=dict(mix_tv_distance=round(_tv(A, B), 6), offset_sigma=float(sigma),
                              n_features_shifted=int(len(off_feats)), novel_share=mag["novel_share"] if novel else 0.0),
        affected_classes=[source.classes[c] for c in affected], affected_features=[source.feature_names[i] for i in off_feats],
        events=events, segments=segments,
        slope_sigma_per_batch=(sigma / spec.incremental_width) if spec.kind == "incremental" else None,
        per_batch=dict(p_b=[round(float(v), 6) for v in batch_p]),
        notes=["synthetic drift: re-sequenced class mix + bounded feature offset; not naturally occurring"])
    return Scenario(spec, stream, truth, p, source_ids)


def scenario_library(seeds_validation: Sequence[int], seeds_test: Sequence[int], kinds=KINDS, magnitudes=MAGNITUDES,
                     controls_per_family: int = 0, control_batches: int = 300, **spec_kwargs) -> List[ScenarioSpec]:
    """Every kind x magnitude x seed, validation and test families kept apart.  ``controls_per_family`` adds that
    many long drift-free 'stationary' streams per family (false-alarm measurement)."""
    out = []
    for fam, seeds in (("validation", seeds_validation), ("test", seeds_test)):
        for sd in list(seeds)[:controls_per_family]:
            out.append(ScenarioSpec(kind=CONTROL_KIND, magnitude="low", seed=sd, family=fam, n_batches=control_batches,
                                    **{k: v for k, v in spec_kwargs.items() if k != "n_batches"}))
    for fam, seeds in (("validation", seeds_validation), ("test", seeds_test)):
        for k in kinds:
            for m in magnitudes:
                for sd in seeds:
                    out.append(ScenarioSpec(kind=k, magnitude=m, seed=sd, family=fam, **spec_kwargs))
    return out


def batch_class_composition(sc: Scenario, n_classes: Optional[int] = None) -> np.ndarray:
    """(n_batches, n_classes) class counts per replay batch (for figures / tests)."""
    s = sc.stream
    n_classes = n_classes or len(s.classes)
    B = sc.spec.n_batches
    idx = np.repeat(np.arange(B), sc.spec.batch_size)
    out = np.zeros((B, n_classes), dtype=np.int64)
    np.add.at(out, (idx, s.y_stream.astype(int)), 1)
    return out
