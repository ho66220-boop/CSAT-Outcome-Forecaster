import numpy as np
import pandas as pd

from suneung import error_model, total_model
from suneung.metrics import grade_from_pct, grade_from_z, z_of


def test_loo_matches_brute_force():
    rng = np.random.default_rng(0)
    x = rng.normal(250, 20, 30)
    y = 10 + 0.9 * x + rng.normal(0, 15, 30)
    fast = total_model.loo_pred(x, y)
    slow = [np.polyval(np.polyfit(np.delete(x, i), np.delete(y, i), 1), x[i]) for i in range(30)]
    assert np.allclose(fast, slow)


def test_sigma_mle_recovers_slope():
    rng = np.random.default_rng(1)
    p = rng.uniform(180, 290, 4000)
    th = np.array([np.log(18), -0.15])
    e = rng.normal(0, np.exp(th[0] + th[1] * (p - 240) / 10))
    est = error_model.fit_sigma(p, e)
    assert abs(est[0] - th[0]) < 0.05 and abs(est[1] - th[1]) < 0.02


def test_sigma_floor_and_cap():
    th = [np.log(20), -0.3]
    s = error_model.sigma(th, [150, 240, 300], floor=12, cap=40)
    assert s[0] == 40 and abs(s[1] - 20) < 1e-9 and s[2] == 12


def test_reach_monotone():
    p = error_model.reach(250, 15, np.array([230, 250, 270]))
    assert p[0] > p[1] > p[2] and abs(p[1] - 0.5) < 1e-9


def test_private_equating_removes_inflation():
    rng = np.random.default_rng(2)
    n = 300

    def cohort(boost, idx):
        a = rng.normal(240, 25, n)
        return pd.DataFrame({"3": a + rng.normal(0, 8, n), "6평": a + rng.normal(0, 8, n),
                             "7": a + boost + rng.normal(0, 8, n)}, index=idx)

    prev, curr = cohort(2, range(n)), cohort(20, range(n, 2 * n))
    eq, info = total_model.private_equating(prev, curr, anchor=["3", "6평"], exams=["7"])
    assert abs(info["7"]["shift"] - 18) < 3
    assert abs((eq["7"] - curr["7"]).mean() + 18) < 3


def test_grade_conversions_agree():
    pct = np.array([99, 96, 95, 89, 88, 60, 59, 4, 3])
    assert list(grade_from_pct(pct)) == [1, 1, 2, 2, 3, 4, 5, 8, 9]
    assert list(grade_from_z(z_of([99, 50, 2]))) == [1, 5, 9]


def test_nested_loo_has_no_leak_from_own_outcome():
    rng = np.random.default_rng(3)
    x = rng.normal(250, 20, 40)
    y = 10 + 0.9 * x + rng.normal(0, 15, 40)
    a = error_model.nested_loo(x, y)
    y2 = y.copy()
    y2[5] += 40
    b = error_model.nested_loo(x, y2)
    assert a[0][5] == b[0][5] and np.allclose(a[1][5], b[1][5]) and a[2][5] == b[2][5]


def test_weight_scan_endpoints_match_comparison():
    rng = np.random.default_rng(4)
    n = 60
    a = rng.normal(240, 25, n)
    W = pd.DataFrame({"3": a + rng.normal(0, 10, n), "6평": a + rng.normal(0, 10, n),
                      "9평": a + rng.normal(0, 10, n), "수능": a + rng.normal(0, 12, n)})
    comp, ci = total_model.compare_inputs(W, n_boot=500)
    ws = total_model.weight_scan(W).set_index("가중치")
    assert ws.loc[0.0, "LOO_RMSE"] == comp.set_index("입력").loc["3~6월", "LOO_RMSE"]
    assert {"비교", "n", "RMSE_차이", "하한_95", "상한_95"} <= set(ci.columns)
