"""과목별 등급 예측과 수능 최저 판정.

1. 국어·수학·탐구는 백분위를 z로 바꾸고, 영어는 등급 그대로 과목별 선형회귀를 합니다.
2. LOO 잔차의 표준편차와 상관행렬로 다변량 정규분포를 만들어 시뮬레이션합니다.
3. 시뮬레이션마다 등급을 매기고(탐구는 두 과목 중 좋은 등급), 최저 유형별 충족 비율을 확률로 씁니다.
4. 병목 과목: 최저를 놓친 시뮬레이션에서, 그 과목만 평소(중앙값) 등급이었다면 충족했을 경우가 가장 많은 과목.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

from .metrics import ZC, grade_from_z, z_of
from .schema import EXAM_ORDER

SUBJ = ["국", "수", "영", "탐1", "탐2"]
PATTERNS = [("2합5", 2, 5), ("2합6", 2, 6), ("3합4", 3, 4), ("3합5", 3, 5), ("3합6", 3, 6),
            ("3합7", 3, 7), ("4합5", 4, 5), ("4합6", 4, 6), ("4합8", 4, 8)]
SUBN = {"국": "국어", "수": "수학", "영": "영어", "탐": "탐구"}


def judge(p, ok=0.8, warn=0.4):
    return "안정" if p >= ok else ("경계" if p >= warn else "위험")


def subject_wide(df, sub):
    """학생 × 시험 표 (4과목 완비 시험만). 국·수는 z, 영어는 등급, 탐구는 두 과목 z 평균."""
    d = df[df["complete"]]
    if sub == "국":
        v = z_of(d["국백"])
    elif sub == "수":
        v = z_of(d["수백"])
    elif sub == "영":
        v = d["영등"].values
    else:
        v = (z_of(d["탐1백"]) + z_of(d["탐2백"])) / 2
    t = pd.DataFrame({"학번": d["학번"].values, "시험명": d["시험명"].values, "v": v}).dropna()
    return t.pivot_table(index="학번", columns="시험명", values="v", aggfunc="first")


def features(df, exams, ids=None, target=None, adj=None):
    """학생별 과목 입력(x)과, target 시험이 있으면 결과(y, 등급)를 만듭니다.
    adj: {(과목, 시험): (c, d)} 이면 그 시험 값을 c + d·값 으로 바꿔 씁니다 (사설 실모 보정)."""
    adj = adj or {}
    m = df[df["시험명"].isin(exams)]
    tg = df[df["시험명"] == target].set_index("학번") if target else None
    rows = []
    for sid, g in m.groupby("학번"):
        if ids is not None and sid not in ids:
            continue
        r = {"학번": sid}

        def a(sub, e, v):
            c, d = adj.get((sub, e), (0.0, 1.0))
            return c + d * v

        for nm, col in (("국", "국백"), ("수", "수백")):
            v = g[g[col].notna()]
            zz = [a(nm, e, z) for e, z in zip(v["시험명"], z_of(v[col]))]
            r[nm + "_x"] = np.mean(zz) if zz else np.nan
        v = g[g["영등"].notna()]
        ee = [a("영", e, x) for e, x in zip(v["시험명"], v["영등"])]
        r["영_x"] = np.mean(ee) if ee else np.nan
        # 탐구: 결과 시험(또는 가장 최근 시험)의 두 과목을 기준으로, 같은 과목 기록의 평균
        if tg is not None and sid in tg.index:
            subs = [tg.at[sid, "탐1"], tg.at[sid, "탐2"]]
        else:
            last = g.assign(o=g["시험명"].map(EXAM_ORDER)).sort_values("o").iloc[-1]
            subs = [last["탐1"], last["탐2"]]
        t = pd.concat([g[["시험명", "탐1", "탐1백"]].set_axis(["e", "s", "p"], axis=1),
                       g[["시험명", "탐2", "탐2백"]].set_axis(["e", "s", "p"], axis=1)])
        t = t[t["p"].notna()]
        for k, sn in enumerate(subs, 1):
            v = t[t["s"] == sn]
            if len(v) == 0:
                v = t
            zz = [a("탐", e, z) for e, z in zip(v["e"], z_of(v["p"]))]
            r[f"탐{k}_x"] = np.mean(zz) if zz else np.nan
        if tg is not None and sid in tg.index:
            q = tg.loc[sid]
            for nm, col in (("국", "국백"), ("수", "수백"), ("탐1", "탐1백"), ("탐2", "탐2백")):
                r[nm + "_y"] = float(z_of(q[col])) if pd.notna(q[col]) else np.nan
            r["영_y"] = q["영등"]
            for nm, col in (("국", "국등"), ("수", "수등"), ("영", "영등"), ("탐1", "탐1등"), ("탐2", "탐2등")):
                r[nm + "_g"] = q[col]
        rows.append(r)
    return pd.DataFrame(rows).set_index("학번") if rows else pd.DataFrame()


def loo_residuals(D):
    """과목별 계수(전체 학습)와 LOO 잔차(학생을 빼고 맞춘 직선으로 그 학생을 예측한 오차)."""
    coef, res = {}, {}
    for s in SUBJ:
        ok = D[[s + "_x", s + "_y"]].dropna()
        coef[s] = np.polyfit(ok[s + "_x"], ok[s + "_y"], 1)
        p = pd.Series(np.nan, index=D.index)
        for i in ok.index:
            t = ok.drop(i)
            p[i] = np.polyval(np.polyfit(t[s + "_x"], t[s + "_y"], 1), ok.at[i, s + "_x"])
        res[s] = D[s + "_y"] - p
    return coef, pd.DataFrame(res)


def spread(res):
    """잔차표에서 과목별 표준편차와 상관행렬."""
    sig = {s: float(np.sqrt(np.nanmean(res[s] ** 2))) for s in SUBJ}
    return sig, res[SUBJ].corr().values


def train(D):
    """과목별 계수, LOO 잔차 표준편차, 잔차 상관행렬."""
    coef, res = loo_residuals(D)
    sig, R = spread(res)
    return coef, sig, R


def mu_of(coef, row):
    return {s: float(np.polyval(coef[s], row[s + "_x"])) for s in SUBJ}


def draws(mu, sig, R, n=6000, seed=0, shift=None):
    """시뮬레이션 등급. 탐구는 두 과목 중 좋은 등급입니다."""
    rg = np.random.default_rng(seed)
    L = np.linalg.cholesky(R + 1e-9 * np.eye(len(SUBJ)))
    Z = rg.standard_normal((n, len(SUBJ))) @ L.T
    shift = shift or {}
    v = {s: mu[s] + shift.get(s, 0.0) + sig[s] * Z[:, j] for j, s in enumerate(SUBJ)}
    return {"국": grade_from_z(v["국"]), "수": grade_from_z(v["수"]),
            "영": np.clip(np.floor(v["영"] + 0.5), 1, 9).astype(int),
            "탐": np.minimum(grade_from_z(v["탐1"]), grade_from_z(v["탐2"]))}


def meet(G, n, k):
    """국·수·영·탐 중 좋은 n개 등급 합이 k 이하인지."""
    arr = np.sort(np.stack([G["국"], G["수"], G["영"], G["탐"]], axis=-1), axis=-1)
    return arr[..., :n].sum(axis=-1) <= k


def pattern_probs(G):
    return {nm: float(meet(G, n, k).mean()) for nm, n, k in PATTERNS}


# ── 병목 과목 정의 3가지 ──
def bott_failure(G, border):
    """채택: 실패한 시뮬레이션마다, 그 과목만 중앙값 등급이었다면 충족했을 경우를 셉니다."""
    med = {k: int(np.median(G[k])) for k in G}
    cnt = {k: 0 for k in G}
    for nm, n, k_ in [p for p in PATTERNS if p[0] in border]:
        fail = ~meet(G, n, k_)
        for k in G:
            H = dict(G)
            H[k] = np.minimum(G[k], med[k])
            cnt[k] += int((fail & meet(H, n, k_)).sum())
    return max(cnt, key=cnt.get)


def bott_one_grade(G, border):
    """비교안 1: 그 과목이 한 등급 떨어질 때 충족 확률이 가장 많이 줄어드는 과목."""
    drop = {}
    for k in G:
        H = dict(G)
        H[k] = np.minimum(9, G[k] + 1)
        drop[k] = sum(meet(G, n, k_).mean() - meet(H, n, k_).mean() for nm, n, k_ in PATTERNS if nm in border)
    return max(drop, key=drop.get)


def bott_one_sigma(mu, sig, R, seed, border, n=6000):
    """비교안 2: 그 과목 실력이 1σ 떨어질 때 충족 확률이 가장 많이 줄어드는 과목."""
    base = pattern_probs(draws(mu, sig, R, n, seed))
    drop = {}
    for k, sh in [("국", {"국": -sig["국"]}), ("수", {"수": -sig["수"]}), ("영", {"영": sig["영"]}),
                  ("탐", {"탐1": -sig["탐1"], "탐2": -sig["탐2"]})]:
        P = pattern_probs(draws(mu, sig, R, n, seed, shift=sh))
        drop[k] = sum(base[nm] - P[nm] for nm in border)
    return max(drop, key=drop.get)


def grade_dist(m, s, eng=False):
    out = []
    for k in range(1, 10):
        if eng:
            lo = -np.inf if k == 1 else k - 0.5
            hi = np.inf if k == 9 else k + 0.5
        else:
            hi = np.inf if k == 1 else ZC[k - 2]
            lo = -np.inf if k == 9 else ZC[k - 1]
        out.append(norm.cdf((hi - m) / s) - norm.cdf((lo - m) / s))
    return np.array(out)


def student_result(sid, row, coef, sig, R, n_sim=6000, ok=0.8, warn=0.4, compare=False):
    mu = mu_of(coef, row)
    seed = int(sid) % 9973
    G = draws(mu, sig, R, n_sim, seed)
    P = pattern_probs(G)
    border = [nm for nm, _, _ in PATTERNS if warn <= P[nm] < ok]
    res = {"min": {k: round(v, 3) for k, v in P.items()}, "bott": ""}
    if border:
        res["bott"] = SUBN[bott_failure(G, border)]
        if compare:
            res["bott_alt"] = {"한 등급 하락": SUBN[bott_one_grade(G, border)],
                               "1σ 하락": SUBN[bott_one_sigma(mu, sig, R, seed, border, n_sim)]}
    gd = {"국어": grade_dist(mu["국"], sig["국"]), "수학": grade_dist(mu["수"], sig["수"]),
          "영어": grade_dist(mu["영"], sig["영"], eng=True)}
    c1, c2 = np.cumsum(grade_dist(mu["탐1"], sig["탐1"])), np.cumsum(grade_dist(mu["탐2"], sig["탐2"]))
    gd["탐구"] = np.diff(np.concatenate([[0], 1 - (1 - c1) * (1 - c2)]))   # 두 과목 독립 가정
    res["gd"] = {k: [round(float(x), 3) for x in v[:6]] for k, v in gd.items()}
    return res


def backtest(D, n_sim=3000, ok=0.8, warn=0.4):
    """전년도 LOO 백테스트. 학생마다 그 학생을 빼고 계수를 다시 맞추고, σ와 상관행렬도
    그 학생의 잔차를 뺀 나머지로 다시 계산한 뒤, 판정 구간별 실제 충족 비율을 봅니다."""
    _, res = loo_residuals(D)
    recs = []
    for sid, r in D.iterrows():
        if any(pd.isna(r[s + "_x"]) or pd.isna(r[s + "_y"]) or pd.isna(r[s + "_g"]) for s in SUBJ):
            continue
        mu = {}
        for s in SUBJ:
            okd = D[[s + "_x", s + "_y"]].dropna().drop(sid, errors="ignore")
            mu[s] = np.polyval(np.polyfit(okd[s + "_x"], okd[s + "_y"], 1), r[s + "_x"])
        sig, R = spread(res.drop(sid))
        P = pattern_probs(draws(mu, sig, R, n_sim, int(sid) % 9973))
        act = {"국": np.array(r["국_g"]), "수": np.array(r["수_g"]), "영": np.array(r["영_g"]),
               "탐": np.array(min(r["탐1_g"], r["탐2_g"]))}
        for nm, n, k in PATTERNS:
            recs.append(dict(학번=sid, 유형=nm, p=P[nm], 판정=judge(P[nm], ok, warn), 실제=bool(meet(act, n, k))))
    return pd.DataFrame(recs)


def private_adjust(prev, curr):
    """과목별 사설 실모 보정식 {(과목, 시험): (c, d)}. 총점과 같은 방식(3~6월 기준 연도 간 보정)을 과목 값에 적용하고,
    3~6월 기록이 없는 학생에게 쓸 '원점수 → 보정값' 직선을 돌려줍니다."""
    from .total_model import private_equating
    adj = {}
    for sub in ("국", "수", "영", "탐"):
        _, info = private_equating(subject_wide(prev, sub), subject_wide(curr, sub), clip=None)
        for e, v in info.items():
            adj[(sub, e)] = (v["map"][1], v["map"][0])
    return adj
