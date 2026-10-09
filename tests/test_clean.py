import numpy as np
import pandas as pd

from suneung import clean
from suneung.schema import SCORE_COLS


def frame(rows):
    base = {c: np.nan for c in SCORE_COLS}
    out = []
    for r in rows:
        d = dict(base, 시험명="6평", 국어="언어와매체", 수학="미적분", 영어="영어", 탐1="물리학1", 탐2="화학1")
        d.update(r)
        out.append(d)
    return pd.DataFrame(out)[SCORE_COLS]


def test_majority_fix():
    d = frame([dict(학번=i, 국표=130, 국백=95, 국등=2) for i in range(1, 4)] + [dict(학번=4, 국표=130, 국백=59, 국등=2)])
    iss = clean.find_issues(d)
    assert len(iss) == 1
    r = iss.iloc[0]
    assert r["학번"] == 4 and r["처리"] == "보정" and r["수정값"] == 95
    fixed = clean.apply_auto_fixes(d, iss)
    assert fixed.loc[3, "국백"] == 95


def test_tie_is_flagged_not_fixed():
    d = frame([dict(학번=1, 국표=120, 국백=80), dict(학번=2, 국표=120, 국백=82)])
    iss = clean.find_issues(d)
    assert set(iss["처리"]) == {"확인 필요"} and len(iss) == 2


def test_reversal_flags_the_culprit():
    rows = [dict(학번=i, 수표=100 + i, 수백=40 + 2 * i) for i in range(1, 8)]
    rows[3]["수백"] = 90                       # 표준점수 104 인데 백분위가 앞뒤보다 훨씬 높음
    iss = clean.find_issues(frame(rows))
    assert list(iss["학번"]) == [4] and iss.iloc[0]["처리"] == "확인 필요"


def test_tamgu_checked_per_subject():
    # 같은 표준점수라도 과목이 다르면 백분위가 달라도 됩니다
    d = frame([dict(학번=1, 탐1="물리학1", 탐1표=65, 탐1백=90), dict(학번=2, 탐1="생명과학1", 탐1표=65, 탐1백=93)])
    assert clean.find_issues(d).empty


def test_merge_versions_keeps_old_value_for_blank():
    old = frame([dict(학번=1, 국표=120, 국백=80), dict(학번=2, 국표=110, 국백=70)])
    new = frame([dict(학번=1, 국표=120, 국백=np.nan), dict(학번=2, 국표=110, 국백=71), dict(학번=3, 국표=100, 국백=50)])
    m, log = clean.merge_versions(new, old)
    m = m.set_index("학번")
    assert m.at[1, "국백"] == 80          # 빈칸은 이전 값으로
    assert m.at[2, "국백"] == 71          # 새 값 우선
    assert 3 in m.index
    assert set(log["내용"]) >= {"새 파일 빈칸을 이전 값으로 채움", "값 변경", "새 파일에서 추가"}
