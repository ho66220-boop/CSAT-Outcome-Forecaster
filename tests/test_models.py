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
