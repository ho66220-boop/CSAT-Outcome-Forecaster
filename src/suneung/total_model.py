"""총점(백분위 합) 예측.

모델은 `수능 = a + b × (입력 시험 평균)` 하나로 고정하고, 어떤 시험을 입력으로 쓸지를 비교합니다.
표본이 작아(학습 70~100명) 자유도를 늘리기보다 입력 선택과 검증에 집중합니다.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import rmse
from .schema import EARLY, PRIVATE, SAT, SEPT


def loo_pred(x, y, deg=1):
    """다항 회귀의 LOO 예측값. 모자 행렬 공식으로 한 번에 계산합니다."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    X = np.vander(x, deg + 1, increasing=True)
    H = X @ np.linalg.pinv(X.T @ X) @ X.T
    e = y - H @ y
    return y - e / (1 - np.diag(H))


def mean_of(W, exams):
    cols = [e for e in exams if e in W.columns]
    return W[cols].mean(axis=1) if cols else pd.Series(np.nan, index=W.index)


def has_any(W, exams):
    cols = [e for e in exams if e in W.columns]
    return W[cols].notna().any(axis=1) if cols else pd.Series(False, index=W.index)


def _frame(W, exams, target, mask=None):
    d = pd.DataFrame({"x": mean_of(W, exams), "y": W[target] if target in W else np.nan})
    if mask is not None:
        d = d[mask.reindex(d.index, fill_value=False)]
    return d.dropna()


def paired_rmse_diff(e_a, e_b, n_boot=4000, seed=0):
    """같은 학생들의 두 오차로 RMSE 차이(A − B)와 학생 단위 부트스트랩 95% 구간을 냅니다."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    rg = np.random.default_rng(seed)
    idx = rg.integers(0, len(e_a), (n_boot, len(e_a)))
    d = np.sqrt((e_a[idx] ** 2).mean(axis=1)) - np.sqrt((e_b[idx] ** 2).mean(axis=1))
    return rmse(e_a) - rmse(e_b), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def compare_inputs(W_prev, target=SAT, n_boot=4000):
    """입력 조합별 LOO RMSE (전년도 수능 기준)와 주요 비교의 부트스트랩 구간.

    '기준선'은 회귀 없이 입력 평균을 그대로 예측값으로 쓴 경우입니다.
    7·8월 + 9평(M2)은 7·8월이나 9평 기록이 있는 전년도 학생 모두로 학습하므로 다른 행과 학생 집단이 다릅니다.
    """
    early = has_any(W_prev, EARLY)
    both = early & W_prev.get(SEPT, pd.Series(np.nan, index=W_prev.index)).notna()
    rows, err = [], {}
    for key, name, kind, exams, mask, deg, who in [
        ("e", "3~6월", "회귀", EARLY, early, 1, "3~6월 기록 있음"),
        ("e9", "3~6월 + 9평", "회귀", EARLY + [SEPT], early, 1, "3~6월 기록 있음"),
        ("e78", "3~8월 + 9평", "회귀", EARLY + PRIVATE + [SEPT], early, 1, "3~6월 기록 있음"),
        ("p9", "7·8월 + 9평", "회귀", PRIVATE + [SEPT], None, 1, "7·8월이나 9평 기록 있음"),
        ("e9q", "3~6월 + 9평, 2차식", "회귀", EARLY + [SEPT], early, 2, "3~6월 기록 있음"),
        ("e9b", "3~6월 + 9평 평균 그대로", "기준선", EARLY + [SEPT], early, 0, "3~6월 기록 있음"),
        ("s", "9평 단독", "회귀", [SEPT], both, 1, "3~6월과 9평 모두 있음"),
        ("sb", "9평 점수 그대로", "기준선", [SEPT], both, 0, "3~6월과 9평 모두 있음"),
        ("e9s", "3~6월 + 9평", "회귀", EARLY + [SEPT], both, 1, "3~6월과 9평 모두 있음"),
    ]:
        d = _frame(W_prev, exams, target, mask)
        e = (d.y - (d.x if deg == 0 else loo_pred(d.x, d.y, deg))).values
        err[key] = pd.Series(e, index=d.index)
        rows.append(dict(입력=name, 방식=kind, 대상=who, n=len(d), LOO_RMSE=rmse(e)))
    cmp_rows = []
    for a, b, label in [("e9", "e", "3~6월 + 9평 − 3~6월"), ("e9", "e78", "3~6월 + 9평 − 3~8월 + 9평"),
                        ("e9q", "e9", "2차식 − 선형"), ("e9", "e9b", "회귀 − 평균 그대로"),
                        ("e9s", "s", "3~6월 + 9평 − 9평 단독"), ("s", "sb", "9평 회귀 − 9평 그대로")]:
        common = err[a].index.intersection(err[b].index)
        dlt, lo, hi = paired_rmse_diff(err[a][common], err[b][common], n_boot)
        cmp_rows.append(dict(비교=label, n=len(common), RMSE_차이=dlt, 하한_95=lo, 상한_95=hi))
    return pd.DataFrame(rows), pd.DataFrame(cmp_rows)


def weight_scan(W_prev, target=SAT, weights=np.round(np.linspace(0, 1, 11), 1)):
    """9평 가중치 실험: x = (1 − w) × 3~6월 평균 + w × 9평 (9평이 없으면 3~6월 평균).
    w를 하나 고르는 것도 학습이므로, 가장 좋은 w의 RMSE는 약간 낙관적입니다."""
    early = has_any(W_prev, EARLY)
    m = mean_of(W_prev, EARLY)
    s9 = W_prev[SEPT] if SEPT in W_prev else pd.Series(np.nan, index=W_prev.index)
    rows = []
    for w in weights:
        x = pd.Series(np.where(s9.notna(), (1 - w) * m + w * s9, m), index=W_prev.index)
        d = pd.DataFrame({"x": x, "y": W_prev[target]})[early].dropna()
        rows.append(dict(가중치=float(w), n=len(d), LOO_RMSE=rmse(d.y - loo_pred(d.x, d.y))))
    d = _frame(W_prev, EARLY + [SEPT], target, early)
    rows.append(dict(가중치="시험 평균 (채택)", n=len(d), LOO_RMSE=rmse(d.y - loo_pred(d.x, d.y))))
    return pd.DataFrame(rows)


def private_equating(W_prev, W_curr, anchor=EARLY, exams=PRIVATE, clip=(0, 300)):
    """사설 실모(7·8월)를 전년도 척도로 옮깁니다.

    시험마다 전년도와 올해 각각 '3~6월 평균 → 그 시험 점수' 직선을 맞추고,
    같은 3~6월 평균에서 올해가 더 높게 나온 만큼을 뺍니다.
    3~6월 기록이 없는 학생(6평 이후 등원)은, 기록 있는 학생에게서 배운 '원점수 → 보정값' 직선을 씁니다.
    """
    m_prev, m_curr = mean_of(W_prev, anchor), mean_of(W_curr, anchor)
    out, info = W_curr.copy(), {}
    for e in exams:
        if e not in W_curr or e not in W_prev:
            continue
        A = pd.DataFrame({"x": m_prev, "y": W_prev[e]}).dropna()
        B = pd.DataFrame({"x": m_curr, "y": W_curr[e]}).dropna()
        fa, fb = np.polyfit(A.x, A.y, 1), np.polyfit(B.x, B.y, 1)
        shift = pd.Series(np.polyval(fb, m_curr) - np.polyval(fa, m_curr), index=W_curr.index)
        by_anchor = W_curr[e] - shift
        d = pd.DataFrame({"raw": W_curr[e], "adj": by_anchor}).dropna()
        mp = np.polyfit(d.raw, d.adj, 1)
        v = pd.Series(np.where(m_curr.notna(), by_anchor, np.polyval(mp, W_curr[e])), index=W_curr.index)
        out[e] = v.clip(*clip) if clip else v
        info[e] = dict(n_prev=len(A), n_curr=len(B), shift=float(shift[B.index].mean()), map=[float(x) for x in mp])
    return out, info


def sept_check(W_prev, W_curr, W_curr_eq, ids_m1, ids_m2):
    """9평 사전 예측 점검.

    9평 이전 시험만으로 9평을 예측하는 모델을 전년도에서 학습하고, 올해 실제 9평으로 채점합니다.
    사설 실모를 그대로 쓸 때(원점수)와 연도 간 보정했을 때(보정), 아예 뺄 때(3~6월)를 비교합니다.
    """
    early = has_any(W_prev, EARLY)
    variants = [
        ("3~6월", EARLY, early, W_curr, ids_m1),
        ("3~8월 원점수", EARLY + PRIVATE, early, W_curr, ids_m1),
        ("3~8월 보정", EARLY + PRIVATE, early, W_curr_eq, ids_m1),
        ("보정 7·8월 (6평 이후 등원)", PRIVATE, has_any(W_prev, PRIVATE), W_curr_eq, ids_m2),
    ]
    rows, preds = [], {}
    for name, exams, mask, W, ids in variants:
        d = _frame(W_prev, exams, SEPT, mask)
        coef = np.polyfit(d.x, d.y, 1)
        ref = rmse(d.y - loo_pred(d.x, d.y))
        Wi = W.loc[W.index.intersection(list(ids))]
        x = mean_of(Wi, exams)
        pred = pd.Series(np.minimum(300, np.polyval(coef, x)), index=Wi.index).dropna()
        preds[name] = (pred, ref)
        act = Wi[SEPT] if SEPT in Wi else pd.Series(dtype=float)
        e = (act - pred).dropna()
        rows.append(dict(방식=name, 학습_n=len(d), 학습_LOO_RMSE=ref, 채점_n=len(e),
                         치우침=float(e.mean()) if len(e) else np.nan, RMSE=rmse(e) if len(e) else np.nan,
                         범위_안=float((e.abs() <= ref).mean()) if len(e) else np.nan))
    return pd.DataFrame(rows), preds


@dataclass
class TotalModel:
    name: str
    exams: list
    coef_p: np.ndarray       # 백분위 합
    coef_t: np.ndarray       # 표준점수 합
    rmse_p: float
    rmse_t: float
    n: int
    pred_loo: np.ndarray     # 학습 학생의 LOO 예측 (오차 모델용)
    y: np.ndarray
    x: np.ndarray            # 학습 학생의 입력 평균

    def predict(self, x):
        x = np.asarray(x, float)
        return np.minimum(300, np.polyval(self.coef_p, x)), np.polyval(self.coef_t, x)


def fit_total(name, Wp, Wt, exams, mask=None, target=SAT):
    d = pd.DataFrame({"x": mean_of(Wp, exams), "yp": Wp[target], "yt": Wt[target]})
    if mask is not None:
        d = d[mask.reindex(d.index, fill_value=False)]
    d = d.dropna()
    pl, tl = loo_pred(d.x, d.yp), loo_pred(d.x, d.yt)
    return TotalModel(name, list(exams), np.polyfit(d.x, d.yp, 1), np.polyfit(d.x, d.yt, 1),
                      rmse(d.yp - pl), rmse(d.yt - tl), len(d), pl, d.yp.values, d.x.values)
