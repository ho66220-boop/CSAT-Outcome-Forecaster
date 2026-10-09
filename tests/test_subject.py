import numpy as np

from suneung import subject_model as sm


def test_meet():
    G = {"국": np.array([1, 3]), "수": np.array([2, 3]), "영": np.array([2, 1]), "탐": np.array([4, 5])}
    assert list(sm.meet(G, 3, 5)) == [True, False]   # 1+2+2=5 / 1+3+3=7
    assert list(sm.meet(G, 2, 3)) == [True, False]


def test_bottleneck_picks_weak_subject():
    rng = np.random.default_rng(0)
    n = 4000
    G = {"국": np.ones(n, int), "수": np.ones(n, int), "영": np.full(n, 2),
         "탐": rng.choice([1, 2, 4], n, p=[0.5, 0.2, 0.3])}   # 탐구만 자주 무너짐
    assert sm.bott_failure(G, ["4합5"]) == "탐"


def test_simulated_probabilities_are_monotone_in_strictness():
    mu = {s: 1.0 for s in sm.SUBJ}
    mu["영"] = 2.0
    sig = {s: 0.4 for s in sm.SUBJ}
    sig["영"] = 0.6
    P = sm.pattern_probs(sm.draws(mu, sig, np.eye(5), n=4000, seed=1))
    assert P["3합4"] <= P["3합5"] <= P["3합6"] <= P["3합7"]
    assert P["4합5"] <= P["4합6"] <= P["4합8"]


def test_grade_dist_sums_to_one():
    assert abs(sm.grade_dist(0.5, 0.4).sum() - 1) < 1e-9
    assert abs(sm.grade_dist(2.3, 0.6, eng=True).sum() - 1) < 1e-9
