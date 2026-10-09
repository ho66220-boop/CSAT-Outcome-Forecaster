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


def compare_inputs(W_prev, target=SAT):
    """입력 조합별 LOO RMSE (전년도 수능 기준)."""
    early = has_any(W_prev, EARLY)
    rows = []
    for name, exams, mask, deg in [
        ("3~6월", EARLY, early, 1),
        ("3~6월 + 9평", EARLY + [SEPT], early, 1),
        ("3~8월 + 9평", EARLY + PRIVATE + [SEPT], early, 1),
        ("7·8월 + 9평", PRIVATE + [SEPT], None, 1),
        ("3~6월 + 9평, 2차식", EARLY + [SEPT], early, 2),
    ]:
        d = _frame(W_prev, exams, target, mask)
        rows.append(dict(입력=name, n=len(d), LOO_RMSE=rmse(d.y - loo_pred(d.x, d.y, deg))))
    # 9평 단독은 9평과 3~6월 기록이 모두 있는 같은 학생끼리 비교합니다
    both = early & W_prev.get(SEPT, pd.Series(np.nan, index=W_prev.index)).notna()
    for name, exams in [("9평 단독", [SEPT]), ("3~6월 + 9평 (9평 단독과 같은 학생)", EARLY + [SEPT])]:
        d = _frame(W_prev, exams, target, both)
        rows.append(dict(입력=name, n=len(d), LOO_RMSE=rmse(d.y - loo_pred(d.x, d.y))))
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
                      rmse(d.yp - pl), rmse(d.yt - tl), len(d), pl, d.yp.values)
