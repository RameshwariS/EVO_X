import pandas as pd

from evox.evaluation.comparison import hypotheses, paired_statistics


def results(seeds=2):
    return pd.DataFrame([dict(system=system, seed=seed, macro_f1=.8 + (.01 if system=="evox" else 0),
                              updates=1, forgetting=0., harmful_promotions=0, training_seconds=.1)
                         for system in ("static","evox","always_retrain") for seed in range(seeds) for scenario in range(10)])


def test_statistics_pair_seeds_not_scenario_rows():
    stats = paired_statistics(results())
    assert (stats.n_pairs == 2).all() and stats.p.isna().all()


def test_identical_seed_differences_and_multiplicity_are_safe():
    stats = paired_statistics(results(6))
    assert stats.p.notna().all() and (stats.holm_p <= 1).all()
    assert (stats[stats.metric == "updates"].p == 1).all()


def test_synthetic_and_quick_results_never_claim_research_success():
    frame = results(6)
    stats = paired_statistics(frame)
    for tag, quick in (("SYNTHETIC", False),("REAL", True)):
        assert {r["status"] for r in hypotheses(frame, stats, tag, quick).values()} == {"inconclusive"}


def test_single_policy_runs_do_not_require_a_comparator():
    frame=results()
    for system in ("evox","static"):
        assert paired_statistics(frame[frame.system==system]).empty
