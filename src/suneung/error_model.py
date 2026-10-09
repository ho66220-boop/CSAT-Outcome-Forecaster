"""점수대별 오차 모델.

    σ(p) = exp(θ0 + θ1 · (p − 240) / 10),  floor ≤ σ ≤ cap

LOO 잔차로 θ를 최대우도 추정합니다. 모든 학생에게 같은 σ를 쓰면 상위권에는 지나치게 넓고
하위권에는 좁습니다. floor는 상위권 σ가 지나치게 작아지는 것을, cap은 아주 낮은 점수에서
σ가 비현실적으로 커지는 것을 막습니다.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

CENTER, SCALE = 240.0, 10.0
DEFAULT_CUTS = (220, 240, 250, 260, 270, 280, 290)


def fit_sigma(pred, resid):
    pred, resid = np.asarray(pred, float), np.asarray(resid, float)

    def nll(th):
        return -np.sum(norm.logpdf(resid, 0, np.exp(th[0] + th[1] * (pred - CENTER) / SCALE)))

    return minimize(nll, [np.log(20.0), 0.0], method="Nelder-Mead").x


def sigma(theta, p, floor=12.0, cap=40.0):
    s = np.exp(theta[0] + theta[1] * (np.asarray(p, float) - CENTER) / SCALE)
    return np.clip(s, floor, cap)


def reach(mu, sd, cut):
    """예측 mu, 오차 sd에서 컷 이상이 나올 확률."""
    return 1 - norm.cdf((cut - np.asarray(mu, float)) / np.asarray(sd, float))


def _scores(p, y):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return float(np.mean((p - y) ** 2)), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def nested_loo(x, y):
    """학생 i마다 i를 완전히 뺀 나머지 학생들로 회귀선을 맞추고, 그 나머지 안에서 다시 LOO 잔차를 구해 θ를 추정합니다.
    i의 실제 결과는 i에게 적용하는 예측값, θ, 단일 σ 어디에도 들어가지 않습니다.
    돌려주는 값: i의 예측값, θ (학생마다), 단일 σ (학생마다)."""
    from .total_model import loo_pred
    x, y = np.asarray(x, float), np.asarray(y, float)
    pred, thetas, uni = np.empty(len(x)), [], np.empty(len(x))
    for i in range(len(x)):
        m = np.arange(len(x)) != i
        pr = loo_pred(x[m], y[m])
        e = y[m] - pr
        thetas.append(fit_sigma(pr, e))
        uni[i] = np.sqrt(np.mean(e ** 2))
        pred[i] = np.polyval(np.polyfit(x[m], y[m], 1), x[i])
    return pred, np.array(thetas), uni


def loo_probs(x, y, floor=12.0, cap=40.0, cuts=DEFAULT_CUTS, nested=None):
    """학생 × 컷별 도달 확률(중첩 LOO)과 실제 도달 여부."""
    pred, th, _ = nested if nested is not None else nested_loo(x, y)
    y = np.asarray(y, float)
    rows = [(i, c, float(reach(pred[i], sigma(th[i], pred[i], floor, cap), c)), float(y[i] >= c))
            for i in range(len(pred)) for c in cuts]
    return pd.DataFrame(rows, columns=["i", "컷", "p", "y"])


def floor_eval(x, y, floors=(0, 8, 10, 12, 14), cap=40.0, cuts=DEFAULT_CUTS, nested=None):
    """σ 모델 비교 (중첩 LOO). 컷은 고정된 7개(220~290)이고, 점수 하나가 학생 수 × 컷 수 건에서 나옵니다.
    '단일 σ'는 모든 학생에게 나머지 학생들의 LOO RMSE를 쓰는 기준선입니다."""
    nested = nested if nested is not None else nested_loo(x, y)
    pred, th, uni = nested
    y = np.asarray(y, float)
    Y = np.array([float(y[i] >= c) for i in range(len(pred)) for c in cuts])
    rows = []
    for f in floors:
        P = loo_probs(x, y, f, cap, cuts, nested)["p"].values
        b, ll = _scores(P, Y)
        rows.append(dict(방식=f"floor {f}", Brier=b, LogLoss=ll))
    P = np.array([reach(pred[i], uni[i], c) for i in range(len(pred)) for c in cuts])
    b, ll = _scores(P, Y)
    rows.append(dict(방식="단일 σ", Brier=b, LogLoss=ll))
    out = pd.DataFrame(rows)
    out.attrs["n_students"], out.attrs["n_cuts"] = len(pred), len(cuts)
    return out


def calibration(x, y, floor=12.0, cap=40.0, cuts=DEFAULT_CUTS, bins=(0, .2, .4, .6, .8, 1.0001), nested=None):
    """예측 확률 구간별 실제 도달 비율 (중첩 LOO 확률 기준)."""
    d = loo_probs(x, y, floor, cap, cuts, nested)
    d["구간"] = pd.cut(d.p, bins, right=False)
    return d.groupby("구간", observed=True).agg(건수=("y", "size"), 평균_예측=("p", "mean"), 실제_비율=("y", "mean")).reset_index()
