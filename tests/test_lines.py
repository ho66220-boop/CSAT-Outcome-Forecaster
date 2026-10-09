import pandas as pd

from suneung import lines

CFG = {"full_types": ["국수영탐", "국수탐"], "exclude_admission_regex": "실기|지역", "min_units": 3,
       "groups": [{"name": "메디컬", "unit_regex": "^(의예|약학)"},
                  {"name": "A", "universities": ["가대"]},
                  {"name": "교대", "unit_contains": "초등교육"}]}


def units():
    rows = [("가대", "경영학과", "인문", 270, "일반전형"), ("가대", "행정학과", "인문", 272, "일반전형"),
            ("가대", "심리학과", "인문", 280, "일반전형"), ("가대", "사회학과", "인문", 200, "지역인재전형"),
            ("가대", "의예과", "자연", 297, "일반전형"), ("나대", "AI신약학과", "자연", 250, "일반전형"),
            ("다교대", "초등교육과", "인자", 247, "일반전형"), ("다교원대", "국어교육과", "인문", 260, "일반전형")]
    b = pd.DataFrame(rows, columns=["대학명", "모집단위명", "계열", "백분위배치컷", "전형명"])
    b["반영유형"], b["탐구수"] = "국수영탐", 2
    return b


def test_group_assignment():
    g = lines.assign_groups(units(), CFG).set_index("모집단위명")["그룹"]
    assert g["의예과"] == "메디컬"
    assert g["AI신약학과"] == ""          # 정규식은 앞에서부터 일치해야 함
    assert g["초등교육과"] == "교대" and g["국어교육과"] == ""
    assert g["경영학과"] == "A"


def test_line_cut_is_median_and_excludes_restricted():
    b = lines.assign_groups(units(), CFG)
    L = {x["g"]: x for x in lines.line_cuts(b, CFG)}
    assert L["A"]["cut"]["인문"] == 272       # 지역인재 200은 제외
    assert L["A"]["cut"]["자연"] is None      # 3개 미만이면 라인 없음


def test_line_stats_counts():
    L = [{"g": "X", "cut": {"인문": 250.0, "자연": None}}]
    st = lines.line_stats([280, 250, 220], [12, 12, 12], L, n_draw=2000)
    r = st.iloc[0]
    assert r["안정"] == 1 and r["가능"] == 1 and 1.4 < r["예상_인원"] < 1.6
