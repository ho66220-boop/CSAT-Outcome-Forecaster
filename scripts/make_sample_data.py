"""가상 학생 데이터 생성기.

실제 학생 데이터 없이 파이프라인 전체를 돌려볼 수 있도록, 실제 파일과 같은 형식의 가상 데이터를 만듭니다.
학생 번호, 대학 이름, 배치컷은 모두 지어낸 값입니다.

만드는 파일 (기본 위치 data/sample/)
  scores_prev.csv   전년도 모의고사와 수능 (모델 학습용)
  scores_curr.csv   올해 3~8월 모의고사
  sept_v1.csv       올해 9월 모의평가 1차 파일
  sept_v2.csv       올해 9월 모의평가 갱신 파일 (추가 학생, 일부 빈칸, 일부 정정)
  roster.csv        재원 명단 (학년, 등원일, 퇴원일)
  placement.csv     대학 배치기준점수표
  fixes.csv         원본 확인 후 수동 보정 목록
  injected_errors.csv  일부러 넣은 입력 오류 (검출 확인용 정답지)

사용법: python scripts/make_sample_data.py --out data/sample --seed 2
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

COLS = ["시험명", "학번", "국어", "국표", "국백", "국등", "수학", "수표", "수백", "수등", "영어", "영등",
        "탐1", "탐1표", "탐1백", "탐1등", "탐2", "탐2표", "탐2백", "탐2등"]
SOC = ["생활과윤리", "사회문화", "윤리와사상", "한국지리", "세계지리", "정치와법", "동아시아사"]
SCI = ["물리학1", "화학1", "생명과학1", "지구과학1", "생명과학2", "지구과학2"]
MONTH = {"3": 0, "4": 1, "5": 2, "6평": 3, "7": 4, "8": 5, "9평": 6, "10": 7, "수능": 8}
GRADE_CUM = np.array([4, 11, 23, 40, 60, 77, 89, 96])
ENG_CUM = np.array([6, 20, 42, 63, 78, 88, 94, 98])   # 영어 절대평가 등급별 누적 비율 가정


def grade_from_pct(p):
    g = np.full(np.shape(p), 9)
    for k, c in enumerate(100 - GRADE_CUM, start=1):
        g = np.where((np.asarray(p) >= c) & (g == 9), k, g)
    return g


def eng_grade(z):
    zc = norm.ppf(1 - ENG_CUM / 100)
    g = np.full(np.shape(z), 9)
    for k in range(8, 0, -1):
        g = np.where(np.asarray(z) >= zc[k - 1], np.minimum(g, k), g)
    return g


class Scales:
    """시험·과목별 표준점수 척도. 같은 시험에서 표준점수가 같으면 백분위·등급도 같도록 만듭니다."""

    def __init__(self, rng):
        self.rng, self.s = rng, {}

    def get(self, cohort, exam, subj):
        key = (cohort, exam, subj)
        if key not in self.s:
            if subj in ("국", "수"):
                self.s[key] = (100 + self.rng.normal(0, 3), 20 + self.rng.uniform(-2, 2), 40, 150)
            else:
                self.s[key] = (50 + self.rng.normal(0, 2), 10 + self.rng.uniform(-1, 1), 20, 80)
        return self.s[key]

    def score(self, cohort, exam, subj, z):
        mu, sd, lo, hi = self.get(cohort, exam, subj)
        std = int(np.clip(round(mu + sd * z), lo, hi))
        pct = int(np.clip(round(100 * norm.cdf((std - mu) / sd)), 0, 100))
        return std, pct, int(grade_from_pct(pct))


def make_students(rng, n, id0, kinds):
    """잠재 실력(g)과 과목별 편차, 성장 속도, 선택과목을 정합니다."""
    rows = []
    for i in range(n):
        g = rng.normal(0.75, 0.55)
        level = "고3" if rng.random() < 0.3 else "N수"
        human = rng.random() < 0.45
        math = "확률과통계" if human else ("미적분" if rng.random() < 0.8 else "기하")
        r = rng.random()
        if human or r < 0.2:
            tams = list(rng.choice(SOC, 2, replace=False))
        elif r < 0.3:
            tams = [str(rng.choice(SCI)), str(rng.choice(SOC))]
        else:
            tams = list(rng.choice(SCI, 2, replace=False))
        rows.append(dict(
            학번=id0 + i, kind=kinds[i], 학년=level, g=g,
            dev={k: rng.normal(0, 0.35) for k in ["국", "수", "영", "탐1", "탐2"]},
            slope=rng.normal(0.03, 0.015) if level == "고3" else rng.normal(0.012, 0.012),
            국어=str(rng.choice(["화법과작문", "언어와매체"])), 수학=math, tams=tams,
            change=rng.random() < 0.08,
        ))
    return rows


def exams_for(kind, cohort, rng):
    if kind == "full":
        ex = [e for e in ["3", "4", "5", "6평", "7", "8", "9평"] if rng.random() < 0.85]
    elif kind == "late":
        ex = [e for e in ["7", "8", "9평"] if rng.random() < 0.9]
    elif kind == "sept":
        ex = ["9평"]
    elif kind == "left":
        ex = [e for e in ["3", "4", "5", "6평"] if rng.random() < 0.85]
    else:
        ex = []
    if cohort == "prev" and kind in ("full", "late"):
        ex = ex + (["10"] if rng.random() < 0.6 else []) + ["수능"]
    return ex


def simulate_scores(students, cohort, rng, scales, private_boost, exam_shift):
    rows = []
    for s in students:
        for e in exams_for(s["kind"], cohort, rng):
            t = MONTH[e]
            if e == "수능":
                sd_noise = 0.18 + 0.45 * max(0.0, 1.1 - s["g"])   # 하위권일수록 수능 변동이 큼
                day = rng.normal(0, 0.10)
            else:
                sd_noise, day = 0.38, 0.0
            boost = private_boost if e in ("7", "8") else 0.0

            def z(k):
                noise = sd_noise * (0.6 if k == "영" else 1.0)   # 영어(절대평가)는 등급 변동이 작음
                return s["g"] + s["dev"][k] + s["slope"] * t + exam_shift[e] + boost + day + rng.normal(0, noise)

            tams = list(s["tams"])
            if s["change"] and t >= 4:
                pool = SOC if tams[1] in SOC else SCI
                tams[1] = [x for x in pool if x not in tams][0]
            r = {"시험명": e, "학번": s["학번"], "국어": s["국어"], "수학": s["수학"], "영어": "영어"}
            r["국표"], r["국백"], r["국등"] = scales.score(cohort, e, "국", z("국"))
            r["수표"], r["수백"], r["수등"] = scales.score(cohort, e, "수", z("수"))
            r["영등"] = int(eng_grade(z("영")))
            for k in (1, 2):
                sub = tams[k - 1]
                r[f"탐{k}"] = sub
                r[f"탐{k}표"], r[f"탐{k}백"], r[f"탐{k}등"] = scales.score(cohort, e, sub, z(f"탐{k}"))
            # 일부 과목 미응시
            if e != "수능":
                for pre in ("국", "수", "탐1", "탐2"):
                    if rng.random() < 0.03:
                        r[pre + "표"] = r[pre + "백"] = r[pre + "등"] = np.nan
            rows.append(r)
    d = pd.DataFrame(rows)[COLS]
    d["_o"] = d["시험명"].map(MONTH)
    return d.sort_values(["학번", "_o"]).drop(columns="_o").reset_index(drop=True)


def inject_errors(curr, rng):
    """입력 실수를 흉내 낸 오류를 넣고, 무엇을 바꿨는지 기록합니다."""
    log = []

    def shared_rows(exam, std_col, sub_col=None, min_n=3):
        d = curr[(curr["시험명"] == exam) & curr[std_col].notna()]
        key = [std_col] + ([sub_col] if sub_col else [])
        cnt = d.groupby(key)[std_col].transform("size")
        return d[cnt >= min_n]

    def unique_rows(exam, std_col, sub_col=None):
        d = curr[(curr["시험명"] == exam) & curr[std_col].notna()]
        key = [std_col] + ([sub_col] if sub_col else [])
        cnt = d.groupby(key)[std_col].transform("size")
        return d[cnt == 1]

    def change(idx, col, new, kind, expect):
        old = curr.at[idx, col]
        curr.at[idx, col] = new
        log.append(dict(학번=int(curr.at[idx, "학번"]), 시험명=curr.at[idx, "시험명"], 열=col,
                        원래값=old, 입력값=new, 유형=kind, 예상처리=expect))

    # 1) 9평 탐구 백분위 오타 (같은 표준점수의 다른 학생이 여럿 있어 다수 값으로 보정 가능)
    r = shared_rows("9평", "탐1표", "탐1").sample(1, random_state=1).index[0]
    p = curr.at[r, "탐1백"]
    change(r, "탐1백", p - 20 if p >= 30 else p + 20, "백분위 오타", "보정")
    # 2) 6평 국어 백분위 오타
    r = shared_rows("6평", "국표").sample(1, random_state=2).index[0]
    p = curr.at[r, "국백"]
    change(r, "국백", p - 5 if p >= 10 else p + 5, "백분위 오타", "보정")
    # 3) 7월 수학 등급 오기
    r = shared_rows("7", "수표").sample(1, random_state=3).index[0]
    g = curr.at[r, "수등"]
    change(r, "수등", g + 1 if g < 9 else g - 1, "등급 오기", "보정")
    # 4) 4월 탐구 백분위 역전 (같은 표준점수가 없어 다수 값을 알 수 없음 → 확인 요청, 원본 확인 후 수동 보정)
    d = unique_rows("4", "탐2표", "탐2")
    d = d[d["탐2백"].between(40, 90)]
    r = d.sample(1, random_state=4).index[0]
    change(r, "탐2백", curr.at[r, "탐2백"] - 25, "백분위 역전", "확인 필요 → 수동 보정")
    # 5) 5월 수학 백분위 역전 (확인 요청으로 남김)
    d = unique_rows("5", "수표")
    d = d[d["수백"].between(20, 70)]
    r = d.sample(1, random_state=5).index[0]
    change(r, "수백", curr.at[r, "수백"] + 25, "백분위 역전", "확인 필요")
    return pd.DataFrame(log)


UNIV_GROUPS = [
    ("최상위권", ["가온대", "나래대", "다솜대"], 287),
    ("상위권", ["라온대", "마루대", "바름대"], 277),
    ("중상위권", ["새봄대", "아라대", "온누리대", "이음대"], 267),
    ("중위권", ["자람대", "차오름대", "큰솔대"], 257),
    ("중하위권", ["타래대", "파랑대", "하람대", "한울대"], 246),
    ("거점국립 A", ["한빛국립대"], 252),
    ("거점국립 B", ["누리국립대"], 246),
]
OTHER_UNIVS = [("들꽃대", 232), ("여울대", 225), ("벼리대", 238)]
HUM = ["경영학과", "경제학과", "국어국문학과", "영어영문학과", "행정학과", "심리학과", "사회학과", "미디어학과"]
NAT = ["기계공학과", "전자공학과", "컴퓨터공학과", "화학과", "생명과학과", "수학과", "건축학과", "간호학과"]


def make_placement(rng):
    rows = []

    def add(univ, unit, track, cut, kind="일반전형", rtype=None, ntam=None):
        rtype = rtype or rng.choice(["국수영탐", "국수영탐", "국수영탐", "국수영탐", "국수탐", "국영탐"])
        if track == "자연" and rtype == "국수영탐" and rng.random() < 0.2:
            rtype = "국수영과"
        ntam = ntam or (1 if rng.random() < 0.1 else 2)
        eng = int(np.clip(round((300 - cut) / 18), 1, 4))
        rows.append(dict(대학명=univ, 모집단위명=unit, 군=str(rng.choice(["가", "나", "다"])), 계열=track,
                         반영유형=rtype, 탐구수=ntam, 백분위배치컷=int(round(cut)), 영어등급=eng, 전형명=kind))

    for _, univs, base in UNIV_GROUPS:
        for u in univs:
            b = base + rng.normal(0, 2)
            for unit in rng.choice(HUM, 5, replace=False):
                add(u, unit, "인문", b - 1 + rng.normal(0, 3))
            for unit in rng.choice(NAT, 5, replace=False):
                add(u, unit, "자연", b + 1 + rng.normal(0, 3))
            add(u, "자유전공학부", "인자", b + rng.normal(0, 2))
            add(u, str(rng.choice(NAT)), "자연", b - 4, kind="지역인재전형")
            add(u, "체육교육과", "예체능", b - 30, kind="실기전형", rtype="국영탐")
    for u, base in OTHER_UNIVS:
        for unit in rng.choice(HUM, 3, replace=False):
            add(u, unit, "인문", base + rng.normal(0, 3))
        for unit in rng.choice(NAT, 3, replace=False):
            add(u, unit, "자연", base + rng.normal(0, 3))
    add("들꽃대", "AI신약학과", "자연", 250, rtype="국수영탐", ntam=2)   # 이름에 '약학'이 있지만 메디컬이 아님
    for u, unit, track, cut in [("가온대", "의예과", "자연", 297), ("나래대", "의예과", "자연", 296),
                                ("라온대", "치의예과", "자연", 293), ("마루대", "한의예과", "자연", 290),
                                ("마루대", "한의예과", "인문", 289), ("새봄대", "약학과", "자연", 291),
                                ("아라대", "수의예과", "자연", 288), ("한빛국립대", "의예과", "자연", 295),
                                ("한빛국립대", "약학과", "인자", 289), ("누리국립대", "의예과", "자연", 294),
                                ("누리국립대", "수의예과", "인문", 286)]:
        add(u, unit, track, cut, rtype="국수영탐", ntam=2)
    for u, cut in [("새솔교대", 248), ("푸른교대", 246), ("한마음교원대", 251)]:
        for kind in ["일반전형", "일반전형", "지역인재전형"]:
            add(u, "초등교육과", "인자", cut + rng.normal(0, 1.5), kind=kind, rtype="국수영탐", ntam=2)
    add("한마음교원대", "국어교육과", "인문", 262, rtype="국수영탐", ntam=2)   # 중등 교육과는 교대 라인에서 제외
    return pd.DataFrame(rows)


def as_int(d):
    d = d.copy()
    for c in d.columns:
        if c[-1] in "표백등" or c in ("원래값", "입력값", "수정값"):
            d[c] = pd.to_numeric(d[c], errors="coerce").round().astype("Int64")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/sample")
    ap.add_argument("--seed", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    scales = Scales(rng)
    exam_shift = {e: rng.normal(0, 0.03) for e in MONTH}

    # 전년도: 연초부터 다닌 학생, 여름에 들어온 학생
    prev_kinds = ["full"] * 85 + ["late"] * 45
    prev = make_students(rng, len(prev_kinds), 800001, prev_kinds)
    sp = simulate_scores(prev, "prev", rng, scales, private_boost=0.05, exam_shift=exam_shift)

    # 올해: 기존 인원, 6평 이후 등원, 9월 등원(9평만), 중도 퇴원, 9평 뒤 퇴원
    kinds = ["full"] * 140 + ["late"] * 32 + ["sept"] * 14 + ["left"] * 12 + ["full_left"] * 6
    curr = make_students(rng, len(kinds), 900001, [k if k != "full_left" else "full" for k in kinds])
    sc = simulate_scores(curr, "curr", rng, scales, private_boost=0.35, exam_shift=exam_shift)
    errors = inject_errors(sc, rng)

    # 9평 파일 두 버전
    sept = sc[sc["시험명"] == "9평"].copy()
    rest = sc[sc["시험명"] != "9평"].copy()
    ids = sept["학번"].unique()
    late_ids = set(rng.choice(ids, int(len(ids) * 0.15), replace=False))
    v2 = sept.copy()
    v1 = sept[~sept["학번"].isin(late_ids)].copy()
    blank_ids = rng.choice(v1["학번"].unique(), 3, replace=False)          # 갱신 파일에서 비어 버린 칸
    err_ids = set(errors["학번"])
    fix_ids = [i for i in rng.choice(v1["학번"].unique(), 6, replace=False) if i not in err_ids and i not in blank_ids][:2]
    v2.loc[v2["학번"].isin(blank_ids), ["탐2", "탐2표", "탐2백", "탐2등"]] = np.nan
    for i in fix_ids:                                                      # 1차 파일의 잘못된 값이 갱신 파일에서 정정됨
        j = v1.index[v1["학번"] == i][0]
        v1.at[j, "국백"] = max(0, v1.at[j, "국백"] - 7)

    # 재원 명단
    def day(lo, hi):
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        return (lo + pd.Timedelta(days=int(rng.integers(0, (hi - lo).days + 1)))).date().isoformat()

    roster = []
    for s, k in zip(curr, kinds):
        enter = {"full": ("2026-01-05", "2026-03-02"), "full_left": ("2026-01-05", "2026-03-02"),
                 "late": ("2026-06-15", "2026-07-20"), "sept": ("2026-08-17", "2026-09-01"),
                 "left": ("2026-01-05", "2026-03-02")}[k]
        leave = {"left": day("2026-06-10", "2026-07-10"), "full_left": day("2026-09-10", "2026-09-30")}.get(k, "")
        roster.append(dict(학번=s["학번"], 학년=s["학년"], 등원일=day(*enter), 퇴원일=leave))

    fixes = errors[errors["예상처리"] == "확인 필요 → 수동 보정"][["학번", "시험명", "열", "원래값"]]
    fixes = fixes.rename(columns={"원래값": "수정값"}).assign(사유="원본 성적표 확인")

    as_int(sp).to_csv(out / "scores_prev.csv", index=False)
    as_int(rest).to_csv(out / "scores_curr.csv", index=False)
    as_int(v1).to_csv(out / "sept_v1.csv", index=False)
    as_int(v2).to_csv(out / "sept_v2.csv", index=False)
    pd.DataFrame(roster).to_csv(out / "roster.csv", index=False)
    make_placement(rng).to_csv(out / "placement.csv", index=False)
    as_int(fixes).to_csv(out / "fixes.csv", index=False)
    as_int(errors).to_csv(out / "injected_errors.csv", index=False)
    print(f"전년도 {sp['학번'].nunique()}명 {len(sp)}행, 올해 {sc['학번'].nunique()}명 {len(sc)}행, "
          f"9평 1차 {len(v1)}행 / 갱신 {len(v2)}행, 넣은 오류 {len(errors)}건 → {out}")


if __name__ == "__main__":
    main()
