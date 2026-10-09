"""대학 라인 정의와 도달 가능성 집계.

라인 = 그룹 안 모집단위들의 백분위 배치컷 중앙값 (계열별).
국어·수학·탐구를 모두 반영하는 모집단위만 쓰고, 실기·지역인재 같은 제한 전형은 뺍니다.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .error_model import reach

TRACKS = {"인문": ["인문", "인자", "인자예"], "자연": ["자연", "인자", "인자예"]}


def load_groups(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assign_groups(B, cfg):
    """모집단위마다 그룹 이름을 붙입니다. 설정 파일의 순서대로 처음 맞는 그룹을 씁니다.
    그룹 조건: unit_regex(모집단위명 정규식, 앞에서부터 일치), unit_contains(모집단위명 포함), universities(대학명 목록)."""
    out = []
    for _, r in B.iterrows():
        g = ""
        for G in cfg["groups"]:
            if "unit_regex" in G and re.match(G["unit_regex"], str(r["모집단위명"])):
                g = G["name"]
            elif "unit_contains" in G and G["unit_contains"] in str(r["모집단위명"]):
                g = G["name"]
            elif r["대학명"] in G.get("universities", []):
                g = G["name"]
            if g:
                break
        out.append(g)
    B = B.copy()
    B["그룹"] = out
    return B


def usable_units(B, cfg):
    """라인 계산과 학과 검색에 쓰는 모집단위: 국·수·탐 전부 반영, 탐구 1~2과목."""
    m = B["반영유형"].isin(cfg["full_types"]) & B["탐구수"].isin([1, 2])
    return B[m].copy()


def line_cuts(B, cfg):
    excl = B["전형명"].astype(str).str.contains(cfg["exclude_admission_regex"])
    lines = []
    for G in cfg["groups"]:
        row = {"g": G["name"], "cut": {}, "n": {}}
        for k, tracks in TRACKS.items():
            x = B[(B["그룹"] == G["name"]) & ~excl & B["계열"].isin(tracks)]["백분위배치컷"].dropna()
            row["cut"][k] = float(np.median(x)) if len(x) >= cfg.get("min_units", 3) else None
            row["n"][k] = int(len(x))
        lines.append(row)
    return lines


def line_stats(mu, sd, lines, n_draw=4000, seed=21, cap=300):
    """집단 집계: 예상 인원(확률 합), 90% 범위(시뮬레이션), 안정(≥80%)·가능(40~80%) 인원."""
    mu, sd = np.asarray(mu, float), np.asarray(sd, float)
    rng = np.random.default_rng(seed)
    draws = np.minimum(cap, mu[None, :] + sd[None, :] * rng.standard_normal((n_draw, len(mu))))
    rows = []
    for L in lines:
        for k in TRACKS:
            c = L["cut"][k]
            if c is None:
                continue
            P = reach(mu, sd, c)
            cnt = (draws >= c).sum(axis=1)
            rows.append(dict(라인=L["g"], 계열=k, 배치컷=c, n=len(mu), 예상_인원=float(P.sum()),
                             하한_5=float(np.percentile(cnt, 5)), 상한_95=float(np.percentile(cnt, 95)),
                             안정=int((P >= .8).sum()), 가능=int(((P >= .4) & (P < .8)).sum())))
    return pd.DataFrame(rows)


def last_year_shares(sat, lines):
    """전년도 실제 수능에서 각 라인 이상을 받은 비율."""
    sat = pd.Series(sat).dropna()
    rows = []
    for L in lines:
        for k in TRACKS:
            c = L["cut"][k]
            if c is not None:
                rows.append(dict(라인=L["g"], 계열=k, 배치컷=c, 전년도_n=len(sat),
                                 전년도_인원=int((sat >= c).sum()), 전년도_비율=float((sat >= c).mean())))
    return pd.DataFrame(rows)


def backcheck(pred_loo, y, sd, lines, track="자연"):
    """전년도 재현 점검: LOO 예측과 σ로 계산한 예상 인원 대 실제 인원."""
    rows = []
    for L in lines:
        c = L["cut"][track]
        if c is not None:
            rows.append(dict(라인=L["g"], 배치컷=c, 예상_인원=float(reach(pred_loo, sd, c).sum()),
                             실제_인원=int((np.asarray(y) >= c).sum())))
    return pd.DataFrame(rows)


def units_for_dashboard(B):
    U = B[B["계열"] != "예체능"]
    cols = ["대학명", "모집단위명", "군", "계열", "반영유형", "탐구수", "백분위배치컷", "영어등급", "전형명"]
    out = []
    for r in U[cols].itertuples(index=False):
        out.append([str(r[0]), str(r[1]), str(r[2]), str(r[3]), str(r[4]), int(r[5]), int(round(r[6])),
                    int(r[7]) if pd.notna(r[7]) else 0, str(r[8])])
    return out
