"""전체 파이프라인: 로드 → 정제 → 대상 정의 → 총점 모델 → 오차 모델 → 대학 라인 → 과목·최저 → 대시보드 → 요약.

단계마다 함수 하나이고, 결과는 ctx(SimpleNamespace)에 모아 다음 단계로 넘깁니다.

ctx 에 담기는 값
  prev, curr        전년도, 올해 성적 (한 행 = 학생 한 명의 시험 한 번)
  roster            재원 명단
  issues            정제 기록 (중복, 수동 보정, 자동 보정, 확인 필요)
  W_prev, T_prev    전년도 학생 × 시험 표 (백분위 합, 표준점수 합). 4과목을 모두 본 시험만
  W_curr            올해 학생 × 시험 백분위 합 표
  W_curr_eq         7·8월을 전년도 척도로 보정한 올해 표
  ids               {"M1": 기존 인원, "M2": 6평 이후 등원, "9평만": ...} 기준일 재원생
  ids_sept          9평 점검 대상 (9평 시점 재원생)
  models            {"M1", "M2"} 총점 모델 (total_model.TotalModel)
  theta, nested     점수대별 σ 계수, 중첩 LOO 결과 (평가용)
  pred              학생별 수능 예상 (구분, 모델, 예상 점수, σ)
  lines             대학 라인 [{g, cut{인문, 자연}, n}]
  subj              학생별 과목 시뮬레이션 결과 (최저 확률, 병목, 등급 분포)
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from . import clean, dashboard, error_model, io, lines, metrics, subject_model, total_model
from .schema import (EARLY, EXAM_ORDER, GROUP_EXISTING, GROUP_LATE, GROUP_SEPT_ONLY, M1_EXAMS, M2_EXAMS,
                     PRE_SEPT, PRIVATE, SAT, SEPT)

GR_EXAMS = PRE_SEPT + [SEPT]
ISSUE_COLS = ["구분", "학번", "시험명", "열", "값", "처리", "수정값", "근거"]


# ── 설정 ──
def load_config(path):
    """설정 파일을 읽고 상대 경로를 저장소 기준(설정 파일 폴더의 상위, 또는 base_dir)으로 풉니다."""
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    base = Path(cfg.get("base_dir", path.parent.parent))

    def fix(v):
        if isinstance(v, dict) and "path" in v:
            return {**v, "path": str((base / v["path"]).resolve())}
        if isinstance(v, str):
            return str((base / v).resolve())
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v

    cfg["data"] = {k: fix(v) for k, v in cfg["data"].items()}
    cfg["groups"] = fix(cfg["groups"])
    d = cfg.setdefault("dashboard", {})
    d["template"] = fix(d.get("template", "templates/dashboard.html"))
    if d.get("font_dir"):
        d["font_dir"] = fix(d["font_dir"])
    return cfg


def settings(cfg, n_sim=None):
    sim, sg, jd = cfg.get("simulation", {}), cfg.get("sigma", {}), cfg.get("judge", {})
    n = n_sim or sim.get("n", 6000)
    return SimpleNamespace(
        n_sim=n, n_bt=min(n, sim.get("backtest_n", 3000)), seed=sim.get("seed", 21),
        floor=sg.get("floor", 12.0), cap=sg.get("cap", 40.0), common=sg.get("common_sd", 0.0),
        ok=jd.get("ok", 0.8), warn=jd.get("warn", 0.4),
        ref=pd.Timestamp(cfg["ref_date"]), sept_date=pd.Timestamp(cfg.get("sept_date", cfg["ref_date"])),
        exclude_unresolved=cfg.get("cleaning", {}).get("exclude_unresolved", True))


# ── 보조 함수 ──
def _scores(entry):
    if isinstance(entry, dict):
        return io.load_scores(entry["path"], entry.get("sheet"))
    return io.load_scores(entry)


def _name(entry):
    return Path(entry if isinstance(entry, str) else entry["path"]).name


def _md_table(df, floatfmt=1):
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v).replace("~", "\\~")      # GitHub 마크다운에서 물결표 두 개가 취소선이 되지 않게
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def _subject_change(g):
    """탐구 두 과목이 모두 적힌 첫 시험과 마지막 시험을 비교해 선택과목 변경을 찾습니다."""
    g = g.assign(o=g["시험명"].map(EXAM_ORDER)).sort_values("o")
    out = []
    t = g[g["탐1"].notna() & g["탐2"].notna()]
    if len(t) >= 2:
        a = sorted({t.iloc[0]["탐1"], t.iloc[0]["탐2"]})
        b = sorted({t.iloc[-1]["탐1"], t.iloc[-1]["탐2"]})
        if a != b:
            out.append(f"탐구 {', '.join(a)} → {', '.join(b)}")
    m = g["수학"].dropna()
    if len(m) >= 2 and m.iloc[0] != m.iloc[-1]:
        out.append(f"수학 {m.iloc[0]} → {m.iloc[-1]}")
    return "; ".join(out)


def _grades(g):
    g = g.set_index("시험명")

    def gv(e, c):
        if e not in g.index or pd.isna(g.at[e, c]):
            return None
        return int(g.at[e, c])

    def gt(e):
        v = [x for x in (gv(e, "탐1등"), gv(e, "탐2등")) if x]
        return min(v) if v else None

    return {"국어": [gv(e, "국등") for e in GR_EXAMS], "수학": [gv(e, "수등") for e in GR_EXAMS],
            "영어": [gv(e, "영등") for e in GR_EXAMS], "탐구": [gt(e) for e in GR_EXAMS]}


def _enrolled(roster, day):
    r = roster[(roster["등원일"] <= day) & (roster["퇴원일"].isna() | (roster["퇴원일"] > day))]
    return set(r["학번"])


# ── 1. 로드 ──
def load_data(ctx, cfg):
    D = cfg["data"]
    prev, d1 = clean.dedupe(_scores(D["prev_scores"]))
    curr, d2 = clean.dedupe(_scores(D["curr_scores"]))
    dups = [d1.assign(구분="전년도"), d2.assign(구분="올해")]
    files = D.get("sept_scores", [])
    files = files if isinstance(files, list) else [files]
    merges = []
    if files:
        sept, d = clean.dedupe(_scores(files[0]))
        dups.append(d.assign(구분="올해"))
        for f in files[1:]:
            new, d = clean.dedupe(_scores(f))
            dups.append(d.assign(구분="올해"))
            sept, ch = clean.merge_versions(new, sept)
            merges.append(ch.assign(파일=_name(f)))
        curr = pd.concat([curr[curr["시험명"] != SEPT], sept], ignore_index=True)
    ctx.prev, ctx.curr = prev, curr
    ctx.dups = pd.concat([d for d in dups if len(d)], ignore_index=True) if any(len(d) for d in dups) else pd.DataFrame()
    ctx.merge_log = pd.concat(merges, ignore_index=True) if merges else pd.DataFrame()
    ctx.merge_log.to_csv(ctx.out / "sept_merge_log.csv", index=False)
    ctx.roster = io.load_roster(D["roster"])
    ctx.log(f"[1] 로드: 전년도 {prev['학번'].nunique()}명, 올해 {curr['학번'].nunique()}명, "
            f"9평 파일 {len(files)}개 병합 (변경 {len(ctx.merge_log)}건), 중복 정리 {len(ctx.dups)}건")


# ── 2. 정제 ──
def clean_data(ctx, cfg):
    curr, manual = clean.apply_manual_fixes(ctx.curr, io.load_fixes(cfg["data"].get("fixes")))
    curr, iss_c = clean.apply_auto_fixes(curr, clean.find_issues(curr), ctx.st.exclude_unresolved)
    prev, iss_p = clean.apply_auto_fixes(ctx.prev, clean.find_issues(ctx.prev), ctx.st.exclude_unresolved)
    parts = [f.assign(구분=k) if "구분" not in f else f
             for f, k in ((ctx.dups, ""), (manual, "올해"), (iss_c, "올해"), (iss_p, "전년도")) if len(f)]
    issues = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=ISSUE_COLS)
    ctx.issues = issues.drop(columns=["row"], errors="ignore").reindex(columns=ISSUE_COLS)
    ctx.issues.to_csv(ctx.out / "data_issues.csv", index=False)
    ctx.prev, ctx.curr = prev, curr
    ctx.log(f"[2] 정제: {ctx.issues['처리'].value_counts().to_dict() if len(ctx.issues) else '문제 없음'}")


# ── 3. 지표와 대상 정의 ──
def define_targets(ctx):
    ctx.prev, ctx.curr = metrics.add_metrics(ctx.prev), metrics.add_metrics(ctx.curr)
    ctx.W_prev, ctx.T_prev = metrics.wide(ctx.prev, "pct3"), metrics.wide(ctx.prev, "total")
    W = ctx.W_curr = metrics.wide(ctx.curr, "pct3")
    early, private = total_model.has_any(W, EARLY), total_model.has_any(W, PRIVATE)
    has9 = W[SEPT].notna() if SEPT in W else pd.Series(False, index=W.index)

    def groups(who):
        m1 = [i for i in W.index if i in who and early[i]]
        m2 = [i for i in W.index if i in who and not early[i] and private[i]]
        s9 = [i for i in W.index if i in who and not early[i] and not private[i] and has9[i]]
        return {"M1": m1, "M2": m2, GROUP_SEPT_ONLY: s9}

    now = _enrolled(ctx.roster, ctx.st.ref)
    ctx.ids = groups(now)
    ctx.ids_sept = groups(_enrolled(ctx.roster, ctx.st.sept_date))   # 9평 점검은 9평 시점 재원생 기준
    # 기준일 재원생인데 어느 집단에도 못 들어간 학생과 그 이유
    placed = set().union(*ctx.ids.values())
    rows_of = set(ctx.curr["학번"])
    ex = [dict(학번=int(i), 사유="올해 성적 기록 없음" if i not in rows_of else "국어, 수학, 탐구 2과목을 모두 본 시험이 없음")
          for i in sorted(now - placed)]
    ctx.excluded = pd.DataFrame(ex, columns=["학번", "사유"])
    ctx.excluded.to_csv(ctx.out / "excluded_students.csv", index=False)
    ctx.log(f"[3] 대상: {GROUP_EXISTING} {len(ctx.ids['M1'])}명, {GROUP_LATE} {len(ctx.ids['M2'])}명, "
            f"{GROUP_SEPT_ONLY} {len(ctx.ids[GROUP_SEPT_ONLY])}명, 제외 {len(ctx.excluded)}명")


# ── 4. 총점 모델 ──
def total_models(ctx):
    ctx.comp, ctx.comp_ci = total_model.compare_inputs(ctx.W_prev)
    ctx.comp.to_csv(ctx.out / "model_comparison.csv", index=False)
    ctx.comp_ci.to_csv(ctx.out / "model_comparison_ci.csv", index=False)
    ctx.wscan = total_model.weight_scan(ctx.W_prev)
    ctx.wscan.to_csv(ctx.out / "sept_weight_scan.csv", index=False)
    ctx.W_curr_eq, eq = total_model.private_equating(ctx.W_prev, ctx.W_curr)
    ctx.sept_tab, ctx.sept_preds = total_model.sept_check(ctx.W_prev, ctx.W_curr, ctx.W_curr_eq,
                                                          ctx.ids_sept["M1"], ctx.ids_sept["M2"])
    ctx.sept_tab.to_csv(ctx.out / "sept_check.csv", index=False)
    ctx.models = {"M1": total_model.fit_total("M1", ctx.W_prev, ctx.T_prev, M1_EXAMS, total_model.has_any(ctx.W_prev, EARLY)),
                  "M2": total_model.fit_total("M2", ctx.W_prev, ctx.T_prev, M2_EXAMS)}
    shifts = ", ".join(f"{e}월 {v['shift']:+.1f}" for e, v in eq.items())
    m1, m2 = ctx.models["M1"], ctx.models["M2"]
    ctx.log(f"[4] 총점 모델: M1 n={m1.n} LOO RMSE {m1.rmse_p:.1f}, M2 n={m2.n} {m2.rmse_p:.1f}; 사설 실모 연도 차 {shifts}")


# ── 5. 오차 모델 ──
def error_models(ctx):
    st = ctx.st
    ctx.theta = {k: error_model.fit_sigma(m.pred_loo, m.y - m.pred_loo) for k, m in ctx.models.items()}
    ctx.nested = {k: error_model.nested_loo(m.x, m.y) for k, m in ctx.models.items()}
    ctx.floor_tab = pd.concat([error_model.floor_eval(m.x, m.y, cap=st.cap, nested=ctx.nested[k]).assign(모델=k, 학생=m.n)
                               for k, m in ctx.models.items()], ignore_index=True)
    ctx.floor_tab.to_csv(ctx.out / "sigma_floor_eval.csv", index=False)
    m1 = ctx.models["M1"]
    ctx.calib = error_model.calibration(m1.x, m1.y, st.floor, st.cap, nested=ctx.nested["M1"])
    ctx.calib.to_csv(ctx.out / "sigma_calibration.csv", index=False)
    ctx.sig_info = {k: {"theta": [float(x) for x in ctx.theta[k]], "floor": st.floor, "cap": st.cap,
                        "sigma_at": {str(p): round(float(error_model.sigma(ctx.theta[k], p, st.floor, st.cap)), 1)
                                     for p in (200, 220, 230, 250, 270)}} for k in ctx.models}
    (ctx.out / "sigma_params.json").write_text(json.dumps(ctx.sig_info, ensure_ascii=False, indent=1), encoding="utf-8")
    ctx.log(f"[5] 오차 모델: M1 σ(200/220/230/250/270) = {list(ctx.sig_info['M1']['sigma_at'].values())}")


# ── 6. 학생별 수능 예상 ──
def predict_students(ctx):
    st, gap, rows = ctx.st, metrics.tam_gap(ctx.curr), []
    for mk, W, exams, grp in (("M1", ctx.W_curr, M1_EXAMS, GROUP_EXISTING), ("M2", ctx.W_curr_eq, M2_EXAMS, GROUP_LATE)):
        ids, m = ctx.ids[mk], ctx.models[mk]
        x = total_model.mean_of(W.loc[ids], exams)
        p, t = m.predict(x)
        sd = error_model.sigma(ctx.theta[mk], p, st.floor, st.cap)
        for i, xi, pi, ti, si in zip(ids, x, p, t, sd):
            rows.append(dict(학번=int(i), 구분=grp, 모델=mk, x=xi, sat_p=pi, sat_t=ti, sd=si,
                             td=m.rmse_t * si / m.rmse_p, gap=float(gap.get(i, 0.0))))
    ctx.pred = pd.DataFrame(rows, columns=["학번", "구분", "모델", "x", "sat_p", "sat_t", "sd", "td", "gap"]).set_index("학번")


# ── 7. 대학 라인 ──
def line_analysis(ctx, cfg):
    st = ctx.st
    gcfg = lines.load_groups(cfg["groups"])
    ctx.units = lines.usable_units(lines.assign_groups(io.load_placement(cfg["data"]["placement"]), gcfg), gcfg)
    L = ctx.lines = lines.line_cuts(ctx.units, gcfg)
    pd.DataFrame([{"라인": x["g"], **{f"{k} 컷": x["cut"][k] for k in x["cut"]}, **{f"{k} 학과 수": x["n"][k] for k in x["n"]}}
                  for x in L]).to_csv(ctx.out / "line_cuts.csv", index=False)
    parts = []
    for grp in (GROUP_EXISTING, GROUP_LATE):
        s = ctx.pred[ctx.pred["구분"] == grp]
        if len(s):
            parts.append(lines.line_stats(s["sat_p"], s["sd"], L, seed=st.seed, common_sd=st.common, ok=st.ok, warn=st.warn).assign(구분=grp))
    stats = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(stats):
        stats = stats.merge(lines.last_year_shares(ctx.W_prev[SAT], L)[["라인", "계열", "전년도_비율"]], on=["라인", "계열"], how="left")
    else:
        ctx.log("    배치컷을 정할 수 있는 라인이 없습니다. 그룹 설정의 min_units 와 대학명을 확인하세요.")
        stats = pd.DataFrame(columns=["라인", "계열", "배치컷", "n", "예상_인원", "하한_5", "상한_95", "안정", "가능", "구분", "전년도_비율"])
    ctx.line_stats = stats
    stats.to_csv(ctx.out / "line_stats.csv", index=False)
    # 공통 이동 민감도: 학생 오차가 함께 움직이면 집단 범위가 얼마나 넓어지는지
    ex, sens = ctx.pred[ctx.pred["구분"] == GROUP_EXISTING], []
    for c in (0.0, 5.0, 10.0):
        t = lines.line_stats(ex["sat_p"], ex["sd"], L, seed=st.seed, common_sd=c) if len(ex) else pd.DataFrame()
        if len(t):
            t = t[t["계열"] == "자연"]
            sens.append(t.assign(공통_σ=c, 범위=t["하한_5"].map("{:.0f}".format) + "~" + t["상한_95"].map("{:.0f}".format)))
    ctx.sens = pd.DataFrame()
    if sens:
        s = pd.concat(sens).pivot_table(index="라인", columns="공통_σ", values="범위", aggfunc="first", sort=False)
        s.columns = [f"공통 σ {c:g}" for c in s.columns]
        ctx.sens = s.reindex([x["g"] for x in L if x["g"] in s.index]).reset_index()
        ctx.sens.to_csv(ctx.out / "line_range_sensitivity.csv", index=False)
    # 전년도 재현 점검: 학생마다 그 학생을 뺀 자료로 만든 예측과 σ (중첩 LOO)
    pr, th, _ = ctx.nested["M1"]
    sd = np.array([error_model.sigma(th[i], pr[i], st.floor, st.cap) for i in range(len(pr))])
    ctx.backcheck = lines.backcheck(pr, ctx.models["M1"].y, sd, L)
    ctx.backcheck.to_csv(ctx.out / "line_backcheck.csv", index=False)
    ctx.log(f"[7] 대학 라인 {len(L)}개, 배치표 {len(ctx.units)}개 모집단위")


# ── 8. 과목 모델과 수능 최저 ──
def subject_analysis(ctx):
    st, prev = ctx.st, ctx.prev
    D1 = subject_model.features(prev, M1_EXAMS, set(ctx.W_prev.index[total_model.has_any(ctx.W_prev, EARLY)]), target=SAT)
    D2 = subject_model.features(prev, M2_EXAMS, set(prev.loc[prev["시험명"].isin(M2_EXAMS), "학번"]), target=SAT)
    D1, D2 = D1[D1["국_g"].notna()], D2[D2["국_g"].notna()]
    T = {"M1": subject_model.train(D1), "M2": subject_model.train(D2)}
    bts = []
    for mk, D in (("M1", D1), ("M2", D2)):
        bt = subject_model.backtest(D, n_sim=st.n_bt, ok=st.ok, warn=st.warn)
        bts.append(subject_model.backtest_summary(bt).assign(모델=mk))
    ctx.backtest = pd.concat(bts, ignore_index=True)
    ctx.backtest.to_csv(ctx.out / "minimum_backtest.csv", index=False)
    adj = subject_model.private_adjust(prev, ctx.curr)
    F = {"M1": subject_model.features(ctx.curr, M1_EXAMS, set(ctx.ids["M1"])),
         "M2": subject_model.features(ctx.curr, M2_EXAMS, set(ctx.ids["M2"]), adj=adj)}
    ctx.subj = {}
    for mk in ("M1", "M2"):
        coef, sg, R = T[mk]
        for sid, r in F[mk].iterrows():
            if any(pd.isna(r[s + "_x"]) for s in subject_model.SUBJ):
                continue
            ctx.subj[int(sid)] = subject_model.student_result(sid, r, coef, sg, R, st.n_sim, st.ok, st.warn, compare=True)
    exist = [int(i) for i in ctx.ids["M1"] if int(i) in ctx.subj]
    ctx.min_table = pd.DataFrame([{"유형": nm, **{j: sum(subject_model.judge(ctx.subj[i]["min"][nm], st.ok, st.warn) == j for i in exist)
                                               for j in ("안정", "경계", "위험")}} for nm, _, _ in subject_model.PATTERNS])
    ctx.min_table.to_csv(ctx.out / "minimum_table.csv", index=False)
    bott = [ctx.subj[i] for i in exist if ctx.subj[i]["bott"]]
    rows = []
    for name, get in (("한 등급 하락 시 충족 확률 감소폭", lambda r: r["bott_alt"]["한 등급 하락"]),
                      ("과목별 1σ 하락 시 감소폭", lambda r: r["bott_alt"]["1σ 하락"]),
                      ("실패 시뮬레이션의 원인 과목 (채택)", lambda r: r["bott"])):
        vc = pd.Series([get(r) for r in bott], dtype=object).value_counts()
        rows.append({"정의": name, **{k: int(vc.get(k, 0)) for k in ("국어", "수학", "영어", "탐구")}})
    ctx.bott_cmp = pd.DataFrame(rows)
    ctx.bott_cmp.to_csv(ctx.out / "bottleneck_compare.csv", index=False)
    ctx.log(f"[8] 과목 모델: 학습 {len(D1)}명/{len(D2)}명, 최저 판정 {len(ctx.subj)}명, 경계 학생 {len(bott)}명")


# ── 9. 대시보드 ──
def build_dashboard(ctx, cfg):
    st, W = ctx.st, ctx.W_curr
    rows = {sid: g for sid, g in ctx.curr.groupby("학번")}
    level = ctx.roster.set_index("학번")["학년"].to_dict()
    students = []
    for sid in ctx.ids["M1"] + ctx.ids["M2"] + ctx.ids[GROUP_SEPT_ONLY]:
        i, g = int(sid), rows[sid]
        mon = {e: (None if (e not in W.columns or pd.isna(W.at[sid, e])) else round(float(W.at[sid, e]), 1)) for e in PRE_SEPT}
        act = W.at[sid, SEPT] if SEPT in W.columns else np.nan
        o = {"id": i, "g": level.get(sid, ""), "mon": mon, "chg": _subject_change(g), "gr": _grades(g)}
        if sid in ctx.pred.index:
            r = ctx.pred.loc[sid]
            # 9평 예측: 9평 이전 시험만으로 전년도 모델을 적용한 값 (기존 인원은 보정한 7·8월까지 포함)
            pr, rr = ctx.sept_preds["3~8월 보정" if r["모델"] == "M1" else "보정 7·8월 (6평 이후 등원)"]
            m = ctx.models[r["모델"]]
            o["grp"] = r["구분"]
            o["n9"] = {"pred": None if sid not in pr.index or pd.isna(pr[sid]) else round(float(pr[sid]), 1),
                       "act": None if pd.isna(act) else round(float(act), 1), "r": round(rr, 1)}
            o["sat"] = {"p": round(float(r["sat_p"]), 1), "t": round(float(r["sat_t"]), 1),
                        "rp": round(m.rmse_p, 1), "rt": round(m.rmse_t, 1),
                        "sd": round(float(r["sd"]), 1), "gap": round(float(r["gap"]), 1), "td": round(float(r["td"]), 1)}
            if i in ctx.subj:
                o.update({k: v for k, v in ctx.subj[i].items() if k != "bott_alt"})
        else:
            o["grp"] = GROUP_SEPT_ONLY
            o["n9"] = {"pred": None, "act": None if pd.isna(act) else round(float(act), 1), "r": None}
        students.append(o)
    meta = {"patterns": [p[0] for p in subject_model.PATTERNS], "date": cfg["ref_date"],
            "judge": {"ok": st.ok, "warn": st.warn},
            "lines": [{"g": x["g"], "cut": x["cut"]} for x in ctx.lines], "units": lines.units_for_dashboard(ctx.units)}
    payload = {"meta": meta, "students": students}
    (ctx.out / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    dc = cfg["dashboard"]
    page = dashboard.render(payload, dc["template"], ctx.out / "dashboard.html", dc.get("title", "수능 예측 대시보드"),
                            dc.get("eyebrow", ""), dc.get("subtitle", ""), dc.get("font_dir"))
    ctx.students = students
    ctx.log(f"[9] 대시보드: {page} ({len(students)}명)")


# ── 10. 학생별 결과표와 요약 ──
def write_summary(ctx, cfg):
    sv = ctx.pred.copy()
    sv["3합6"] = [ctx.subj.get(int(i), {}).get("min", {}).get("3합6") for i in sv.index]
    sv["병목"] = [ctx.subj.get(int(i), {}).get("bott", "") for i in sv.index]
    sv.round(1).to_csv(ctx.out / "students.csv")
    ids, fl = ctx.ids, ctx.floor_tab
    exc = ctx.excluded["사유"].value_counts().rename_axis("사유").reset_index(name="학생") if len(ctx.excluded) else pd.DataFrame()
    lines_ = [
        "# 실행 요약", "",
        f"기준일 {cfg['ref_date']}. 대상 {GROUP_EXISTING} {len(ids['M1'])}명, {GROUP_LATE} {len(ids['M2'])}명, "
        f"{GROUP_SEPT_ONLY} {len(ids[GROUP_SEPT_ONLY])}명. 재원 중이지만 제외한 학생 {len(ctx.excluded)}명.", "",
        "## 제외한 재원생 (excluded_students.csv)", "", _md_table(exc) if len(exc) else "없음", "",
        "## 입력 조합별 LOO RMSE (전년도 수능)", "", _md_table(ctx.comp), "",
        "## 주요 비교의 RMSE 차이 (학생 단위 부트스트랩 95% 구간)", "", _md_table(ctx.comp_ci, 2), "",
        "## 9평 가중치 실험", "", _md_table(ctx.wscan, 2), "",
        f"## 9평 사전 예측 점검 (9평 시점 재원생, 치우침 = 실제 − 예측)", "", _md_table(ctx.sept_tab, 2), "",
        f"## 오차 모델 비교 (중첩 LOO, 학생 × 고정 컷 {fl.attrs.get('n_cuts', 7)}개)", "", _md_table(fl, 4), "",
        "## 보정도 (M1, 중첩 LOO 확률)", "", _md_table(ctx.calib, 3), "",
        "## 점수대별 σ", "", _md_table(pd.DataFrame([{"모델": k, **v["sigma_at"]} for k, v in ctx.sig_info.items()])), "",
        "## 대학 라인별 전망", "", _md_table(ctx.line_stats[ctx.line_stats["구분"] == GROUP_EXISTING].drop(columns=["구분"]), 2), "",
        f"## 공통 이동을 넣었을 때 90% 범위 ({GROUP_EXISTING}, 자연 컷)", "",
        "학생 오차가 서로 독립이면(공통 σ 0) 범위가 좁게 나옵니다. 공통 σ는 1개 연도 자료로 추정할 수 없어 예시 값입니다.", "",
        _md_table(ctx.sens) if len(ctx.sens) else "", "",
        "## 전년도 재현 점검 (자연 컷, 중첩 LOO)", "", _md_table(ctx.backcheck), "",
        "## 수능 최저 백테스트 (전년도, 중첩 LOO, 건수 = 학생 × 9개 유형)", "", _md_table(ctx.backtest, 3), "",
        f"## 수능 최저 판정 ({GROUP_EXISTING})", "", _md_table(ctx.min_table), "",
        "## 병목 과목 정의 비교", "", _md_table(ctx.bott_cmp), "",
        "## 데이터 정제", "", _md_table(ctx.issues) if len(ctx.issues) else "문제 없음", "",
    ]
    (ctx.out / "summary.md").write_text("\n".join(lines_), encoding="utf-8")


def run(config, out_dir="outputs", n_sim=None, log=print):
    cfg = load_config(config) if not isinstance(config, dict) else config
    ctx = SimpleNamespace(out=Path(out_dir), log=log, st=settings(cfg, n_sim))
    ctx.out.mkdir(parents=True, exist_ok=True)
    load_data(ctx, cfg)
    clean_data(ctx, cfg)
    define_targets(ctx)
    total_models(ctx)
    error_models(ctx)
    predict_students(ctx)
    line_analysis(ctx, cfg)
    subject_analysis(ctx)
    build_dashboard(ctx, cfg)
    write_summary(ctx, cfg)
    log(f"[10] 결과 파일: {ctx.out}")
    return {"students": ctx.students, "lines": ctx.lines, "line_stats": ctx.line_stats, "issues": ctx.issues,
            "sept_check": ctx.sept_tab, "model_comparison": ctx.comp, "model_comparison_ci": ctx.comp_ci,
            "floor_eval": ctx.floor_tab, "backtest": ctx.backtest, "bottleneck": ctx.bott_cmp,
            "excluded": ctx.excluded, "weight_scan": ctx.wscan}
