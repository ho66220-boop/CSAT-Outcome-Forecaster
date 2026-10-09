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


def _toy(n=30, seed=0):
    import pandas as pd
    rg = np.random.default_rng(seed)
    D = pd.DataFrame(index=pd.Index(range(1000, 1000 + n), name="학번"))
    for s in sm.SUBJ:
        x = rg.normal(1.0 if s != "영" else 2.5, 0.5, n)
        D[s + "_x"], D[s + "_y"] = x, x + rg.normal(0, 0.4, n)
    for s in sm.SUBJ:
        D[s + "_g"] = rg.integers(1, 6, n)
    return D


def test_backtest_has_no_leak_from_own_outcome():
    """평가 대상 학생의 실제 결과를 바꿔도 그 학생의 예측 확률은 그대로여야 합니다 (중첩 LOO)."""
    D = _toy()
    a = sm.backtest(D, n_sim=500)
    D2 = D.copy()
    sid = D.index[3]
    for s in sm.SUBJ:
        D2.at[sid, s + "_y"] = D.at[sid, s + "_y"] + 3.0
    b = sm.backtest(D2, n_sim=500)
    pa = a[a["학번"] == sid].set_index("유형")["p"]
    pb = b[b["학번"] == sid].set_index("유형")["p"]
    assert (pa == pb).all()
    assert not (a[a["학번"] != sid]["p"].values == b[b["학번"] != sid]["p"].values).all()


def test_nearest_corr_is_positive_definite():
    R = np.array([[1, .9, .9], [.9, 1, -.9], [.9, -.9, 1]])     # 쌍별 상관에서 나올 수 있는 비양정치 행렬
    assert np.linalg.eigvalsh(R).min() < 0
    C = sm.nearest_corr(R)
    assert np.linalg.eigvalsh(C).min() > 0 and np.allclose(np.diag(C), 1)
    np.linalg.cholesky(C)


def test_grade_distribution_keeps_low_grades():
    import pandas as pd
    D = _toy()
    coef, sig, R = sm.train(D)
    row = D.iloc[0].copy()
    row["영_x"] = 8.0                                            # 영어 8등급 수준 학생
    res = sm.student_result(1, row, coef, sig, R, n_sim=2000)
    eng = res["gd"]["영어"]
    assert len(eng) == 9 and abs(sum(eng) - 1) < 0.01 and int(np.argmax(eng)) + 1 >= 7


def test_features_do_not_use_future_subject_history():
    """수능에서 탐구 과목을 바꾼 학생은 바꾸기 전(입력 시험) 과목 기록으로 입력을 만듭니다."""
    import pandas as pd
    from suneung.schema import SCORE_COLS
    rows = []
    for e, sub2, p2 in (("3", "화학1", 50), ("6평", "화학1", 52), ("9평", "화학1", 54), ("수능", "지구과학1", 95)):
        r = {c: np.nan for c in SCORE_COLS}
        r.update(시험명=e, 학번=1, 국백=80, 수백=80, 영등=2, 탐1="물리학1", 탐1백=70, 탐2=sub2, 탐2백=p2,
                 국등=3, 수등=3, 탐1등=4, 탐2등=1 if e == "수능" else 5)
        rows.append(r)
    df = pd.DataFrame(rows)
    F = sm.features(df, ["3", "6평", "9평"], target="수능")
    from suneung.metrics import z_of
    assert F.loc[1, "탐2_x"] == pytest_approx(np.mean(z_of([50, 52, 54])))


def pytest_approx(v):
    import pytest
    return pytest.approx(v, abs=1e-9)
