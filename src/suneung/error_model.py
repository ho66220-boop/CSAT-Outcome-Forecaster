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


def floor_eval(pred, y, floors=(0, 8, 10, 12, 14), cap=40.0, cuts=DEFAULT_CUTS):
    """LOO로 σ 모델을 비교합니다. 학생 i를 빼고 θ를 맞춘 뒤, i의 컷별 도달 확률을 실제와 비교합니다.
    '단일 σ'는 모든 학생에게 나머지 학생의 RMSE를 쓰는 기준선입니다."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    e = y - pred
    P = {f"floor {f}": [] for f in floors}
    P["단일 σ"] = []
    Y = []
    for i in range(len(pred)):
        m = np.arange(len(pred)) != i
        th = fit_sigma(pred[m], e[m])
        for c in cuts:
            Y.append(float(y[i] >= c))
            for f in floors:
                P[f"floor {f}"].append(reach(pred[i], sigma(th, pred[i], f, cap), c))
            P["단일 σ"].append(reach(pred[i], np.sqrt(np.mean(e[m] ** 2)), c))
    Y = np.array(Y)
    rows = []
    for k, v in P.items():
        b, ll = _scores(np.array(v, float), Y)
        rows.append(dict(방식=k, Brier=b, LogLoss=ll))
    return pd.DataFrame(rows)


def calibration(pred, y, th, floor=12.0, cap=40.0, cuts=DEFAULT_CUTS, bins=(0, .2, .4, .6, .8, 1.0001)):
    """예측 확률 구간별 실제 도달 비율."""
    pred, y = np.asarray(pred, float), np.asarray(y, float)
    rows = [(reach(p, sigma(th, p, floor, cap), c), float(t >= c)) for p, t in zip(pred, y) for c in cuts]
    d = pd.DataFrame(rows, columns=["p", "y"])
    d["구간"] = pd.cut(d.p, bins, right=False)
    return d.groupby("구간", observed=True).agg(n=("y", "size"), 평균_예측=("p", "mean"), 실제_비율=("y", "mean")).reset_index()
