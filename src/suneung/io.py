"""파일 읽기. CSV와 엑셀(xlsx)을 모두 받습니다."""
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import MATH_ALIASES, NUMERIC_COLS, ROMAN, SCORE_COLS, SUBJECT_ALIASES

EXAM_ALIASES = {"11수능": "수능", "6월": "6평", "9월": "9평", "6": "6평", "9": "9평"}


def read_table(path, sheet=None):
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path, sheet_name=sheet or 0)
    return pd.read_csv(path)


def norm_subject(s):
    if not isinstance(s, str):
        return s
    s = s.strip()
    for a, b in ROMAN.items():
        s = s.replace(a, b)
    return SUBJECT_ALIASES.get(s, s)


def load_scores(path, sheet=None):
    """성적 파일을 표준 형식으로 읽습니다.

    열 이름이 SCORE_COLS와 같으면 이름으로 고르고(반, 제2외국어 같은 추가 열은 무시),
    원본 엑셀처럼 머리글이 '표준점수', '백분위'로 반복되는 20열 파일은 위치로 맞춥니다.
    점수 0은 미응시로 보고 빈칸으로 바꿉니다.
    """
    d = read_table(path, sheet)
    if set(SCORE_COLS) <= set(d.columns):
        d = d[SCORE_COLS].copy()
    elif d.shape[1] == len(SCORE_COLS):
        d = d.set_axis(SCORE_COLS, axis=1)
    else:
        raise ValueError(f"{path}: 열 구성을 알 수 없습니다 ({d.shape[1]}열). data/README.md 의 형식을 확인하세요.")
    d["시험명"] = d["시험명"].astype(str).str.strip().replace(EXAM_ALIASES)
    d = d[d["학번"].notna()].copy()
    d["학번"] = pd.to_numeric(d["학번"], errors="coerce").astype("Int64")
    for c in NUMERIC_COLS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
        d.loc[d[c] == 0, c] = np.nan
    for c in ("탐1", "탐2"):
        d[c] = d[c].map(norm_subject)
    d["수학"] = d["수학"].replace(MATH_ALIASES)
    return d.reset_index(drop=True)


def load_roster(path, sheet=None):
    r = read_table(path, sheet)
    r["학번"] = pd.to_numeric(r["학번"], errors="coerce").astype("Int64")
    for c in ("등원일", "퇴원일"):
        r[c] = pd.to_datetime(r[c], errors="coerce")
    return r


def load_placement(path, sheet=None):
    b = read_table(path, sheet)
    b["탐구수"] = pd.to_numeric(b["탐구수"], errors="coerce")
    b["백분위배치컷"] = pd.to_numeric(b["백분위배치컷"], errors="coerce")
    return b


def load_fixes(path):
    """수동 보정 목록: 학번, 시험명, 열, 수정값(빈칸이면 해당 값을 지표에서 제외), 사유."""
    if not path or not Path(path).exists():
        return pd.DataFrame(columns=["학번", "시험명", "열", "수정값", "사유"])
    f = read_table(path)
    f["학번"] = pd.to_numeric(f["학번"], errors="coerce").astype("Int64")
    f["시험명"] = f["시험명"].astype(str)
    return f
