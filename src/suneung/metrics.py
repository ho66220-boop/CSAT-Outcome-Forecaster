"""지표 계산: 백분위 합(300점), 표준점수 합, z 변환, 등급 변환."""
import numpy as np
from scipy.stats import norm

# 상대평가 등급 누적 비율(%) 4, 11, 23, 40, 60, 77, 89, 96 에 해당하는 z 하한 (1등급부터)
GRADE_CUM = np.array([4, 11, 23, 40, 60, 77, 89, 96])
ZC = norm.ppf(1 - GRADE_CUM / 100)


def z_of(p):
    """백분위를 표준정규 z 로 바꿉니다. 0과 100은 무한대가 되므로 0.5~99.5로 자릅니다."""
    return norm.ppf(np.clip(np.asarray(p, dtype=float), 0.5, 99.5) / 100)


def grade_from_z(z):
    z = np.asarray(z, dtype=float)
    g = np.full(z.shape, 9, dtype=int)
    for k in range(8, 0, -1):
        g = np.where(z >= ZC[k - 1], np.minimum(g, k), g)
    return g


def grade_from_pct(p):
    """백분위로 상대평가 등급을 매깁니다 (1등급: 백분위 96 이상)."""
    p = np.asarray(p, dtype=float)
    g = np.full(p.shape, 9, dtype=int)
    for k, c in enumerate(100 - GRADE_CUM, start=1):
        g = np.where((p >= c) & (g == 9), k, g)
    return g


def add_metrics(df):
    """complete(국·수·탐2 모두 응시), pct3(백분위 합), total(표준점수 합) 열을 붙입니다."""
    d = df.copy()
    std = d[["국표", "수표", "탐1표", "탐2표"]]
    d["complete"] = (std.fillna(0) > 0).all(axis=1)
    d["pct3"] = d["국백"] + d["수백"] + (d["탐1백"] + d["탐2백"]) / 2
    d["total"] = std.sum(axis=1, min_count=4)
    d.loc[~d["complete"], ["pct3", "total"]] = np.nan
    return d


def wide(df, value="pct3"):
    """학생 × 시험 표. 4과목을 모두 응시한 시험만 씁니다."""
    d = df[df["complete"] & df[value].notna()]
    return d.pivot_table(index="학번", columns="시험명", values=value, aggfunc="first")


def tam_gap(df):
    """탐구 상위 1과목 보정값: 시험별 |탐1 백분위 − 탐2 백분위| / 2 의 학생 평균."""
    t = df.dropna(subset=["탐1백", "탐2백"])
    return ((t["탐1백"] - t["탐2백"]).abs() / 2).groupby(t["학번"]).mean()


def rmse(e):
    e = np.asarray(e, dtype=float)
    return float(np.sqrt(np.nanmean(e ** 2)))
