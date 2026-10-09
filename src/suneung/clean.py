"""입력 오류 검출과 파일 버전 병합.

검출 규칙
  1. 같은 시험, 같은 과목, 같은 표준점수인데 백분위나 등급이 다른 행
     → 같은 표준점수 안에서 한 값이 확실한 다수이면 그 값으로 보정, 아니면 확인 필요
  2. 표준점수가 높은데 백분위가 낮거나 등급이 나쁜 역전
     → 역전을 가장 많이 일으키는 표준점수부터 하나씩 확인 필요로 표시
"""
import numpy as np
import pandas as pd

from .schema import SCORE_COLS

SLOTS = [("국", "국어"), ("수", "수학"), ("탐1", None), ("탐2", None)]


def long_scores(df):
    """국어·수학·탐구 칸을 (시험, 과목, 표준점수, 백분위, 등급) 긴 형식으로 펼칩니다.
    국어·수학은 선택과목과 관계없이 한 척도이고, 탐구는 과목별로 따로 봅니다."""
    parts = []
    for pre, fixed in SLOTS:
        p = pd.DataFrame({
            "row": df.index, "학번": df["학번"], "시험명": df["시험명"],
            "과목": fixed if fixed else df[pre],
            "std": df[pre + "표"], "pct": df[pre + "백"], "grade": df[pre + "등"],
            "std_col": pre + "표", "pct_col": pre + "백", "grade_col": pre + "등",
        })
        parts.append(p[p["std"].notna()])
    return pd.concat(parts, ignore_index=True)


def _duplicate_std(g, val):
    out = []
    for std, h in g.groupby("std"):
        vals = h[val].dropna()
        if vals.nunique() <= 1:
            continue
        vc = vals.value_counts()
        clear = vc.iloc[0] > vc.iloc[1:].sum()
        for _, r in h.iterrows():
            if clear and r[val] == vc.index[0]:
                continue
            out.append(dict(row=r["row"], 학번=r["학번"], 시험명=r["시험명"], 과목=r["과목"], 열=r[val + "_col"],
                            표준점수=std, 값=r[val],
                            처리="보정" if clear else "확인 필요",
                            수정값=vc.index[0] if clear else np.nan,
                            근거=f"같은 표준점수 {len(vals)}명 중 {vc.iloc[0]}명이 {vc.index[0]:g}"))
    return out


def _reversals(g, val):
    """대표값(최빈값) 기준으로 역전 쌍을 세고, 가장 많이 얽힌 표준점수부터 제거합니다."""
    sign = 1 if val == "pct" else -1                      # 백분위는 표준점수와 같은 방향, 등급은 반대
    rep = g.groupby("std")[val].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan).dropna()
    rep = rep.sort_index() * sign
    bad, out = [], []
    while len(rep) > 2:
        v = rep.values
        inv = (v[:, None] > v[None, :]) & (np.arange(len(v))[:, None] < np.arange(len(v))[None, :])
        cnt = inv.sum(axis=0) + inv.sum(axis=1)
        if cnt.max() == 0:
            break
        k = int(cnt.argmax())
        bad.append(rep.index[k])
        rep = rep.drop(rep.index[k])
    for std in bad:
        lo = rep[rep.index < std].max() * sign if (rep.index < std).any() else np.nan
        hi = rep[rep.index > std].min() * sign if (rep.index > std).any() else np.nan
        rng = sorted([x for x in (lo, hi) if pd.notna(x)])
        for _, r in g[g["std"] == std].iterrows():
            what = "백분위" if val == "pct" else "등급"
            out.append(dict(row=r["row"], 학번=r["학번"], 시험명=r["시험명"], 과목=r["과목"], 열=r[val + "_col"],
                            표준점수=std, 값=r[val], 처리="확인 필요", 수정값=np.nan,
                            근거=f"표준점수 순서와 {what}가 역전 (앞뒤 값 {'~'.join(f'{x:g}' for x in rng)})"))
    return out


def find_issues(df):
    L = long_scores(df)
    out = []
    for _, g in L.groupby(["시험명", "과목"]):
        for val in ("pct", "grade"):
            dup = _duplicate_std(g, val)
            out += dup
            # 다수 값으로 보정한 뒤의 값으로 역전을 봅니다
            g2 = g.copy()
            for d in dup:
                if d["처리"] == "보정":
                    g2.loc[g2["row"] == d["row"], val] = d["수정값"]
            out += _reversals(g2, val)
    cols = ["row", "학번", "시험명", "과목", "열", "표준점수", "값", "처리", "수정값", "근거"]
    res = pd.DataFrame(out, columns=cols)
    return res.drop_duplicates(subset=["row", "열"]).reset_index(drop=True)


def apply_auto_fixes(df, issues):
    d = df.copy()
    for _, r in issues[issues["처리"] == "보정"].iterrows():
        d.at[r["row"], r["열"]] = r["수정값"]
    return d


def apply_manual_fixes(df, fixes):
    """수동 보정 목록을 적용하고 적용 내역을 돌려줍니다. 수정값이 빈칸이면 그 값을 비웁니다."""
    d, log = df.copy(), []
    for _, f in fixes.iterrows():
        m = (d["학번"] == f["학번"]) & (d["시험명"].astype(str) == str(f["시험명"]))
        if not m.any():
            log.append(dict(학번=f["학번"], 시험명=f["시험명"], 열=f["열"], 처리="수동 보정 대상 없음", 근거=f.get("사유", "")))
            continue
        for i in d.index[m]:
            old = d.at[i, f["열"]]
            d.at[i, f["열"]] = f["수정값"] if pd.notna(f["수정값"]) else np.nan
            log.append(dict(row=i, 학번=f["학번"], 시험명=f["시험명"], 열=f["열"], 값=old, 수정값=f["수정값"],
                            처리="수동 보정" if pd.notna(f["수정값"]) else "수동 제외", 근거=f.get("사유", "")))
    return d, pd.DataFrame(log)


def merge_versions(new, old, key=("학번", "시험명")):
    """새 파일 값을 우선하고, 새 파일에서 비어 있는 칸은 이전 파일 값으로 채웁니다.
    바뀐 칸을 모두 기록해 돌려줍니다."""
    key = list(key)
    n, o = new.set_index(key), old.set_index(key)
    merged = n.combine_first(o)
    log = []
    for idx in merged.index:
        if idx not in o.index:
            log.append(dict(학번=idx[0], 시험명=idx[1], 열="", 이전="", 새값="", 내용="새 파일에서 추가"))
            continue
        if idx not in n.index:
            log.append(dict(학번=idx[0], 시험명=idx[1], 열="", 이전="", 새값="", 내용="이전 파일에만 있음 (유지)"))
            continue
        for c in n.columns:
            a, b = o.at[idx, c], n.at[idx, c]
            if pd.isna(b) and pd.notna(a):
                log.append(dict(학번=idx[0], 시험명=idx[1], 열=c, 이전=a, 새값="", 내용="새 파일 빈칸을 이전 값으로 채움"))
            elif pd.notna(a) and pd.notna(b) and a != b:
                log.append(dict(학번=idx[0], 시험명=idx[1], 열=c, 이전=a, 새값=b, 내용="값 변경"))
    out = merged.reset_index()
    return out[[c for c in SCORE_COLS if c in out.columns]], pd.DataFrame(log, columns=["학번", "시험명", "열", "이전", "새값", "내용"])
