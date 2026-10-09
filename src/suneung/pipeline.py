"""전체 파이프라인: 로드 → 정제 → 지표 → 총점 모델 → 오차 모델 → 대학 라인 → 과목·최저 → 대시보드."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, dashboard, error_model, io, lines, metrics, subject_model, total_model
from .schema import (EARLY, EXAM_ORDER, GROUP_EXISTING, GROUP_LATE, GROUP_SEPT_ONLY, M1_EXAMS, M2_EXAMS,
                     PRE_SEPT, PRIVATE, SAT, SEPT)

GR_EXAMS = PRE_SEPT + [SEPT]


def load_config(path):
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


def _scores(entry):
    if isinstance(entry, dict):
        return io.load_scores(entry["path"], entry.get("sheet"))
    return io.load_scores(entry)


def _md_table(df, floatfmt=1):
    def f(v):
        if isinstance(v, float):
            return "" if np.isnan(v) else f"{v:.{floatfmt}f}"
        return str(v)
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
            out.append(f"탐구 {'·'.join(a)} → {'·'.join(b)}")
    m = g["수학"].dropna()
    if len(m) >= 2 and m.iloc[0] != m.iloc[-1]:
        out.append(f"수학 {m.iloc[0]} → {m.iloc[-1]}")
    return ", ".join(out)


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


def run(config, out_dir="outputs", n_sim=None, log=print):
    cfg = load_config(config) if not isinstance(config, dict) else config
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    D = cfg["data"]
    sim = cfg.get("simulation", {})
    n_sim = n_sim or sim.get("n", 6000)
    n_bt = min(n_sim, sim.get("backtest_n", 3000))
    floor, cap = cfg.get("sigma", {}).get("floor", 12.0), cfg.get("sigma", {}).get("cap", 40.0)
    ok, warn = cfg.get("judge", {}).get("ok", 0.8), cfg.get("judge", {}).get("warn", 0.4)
    ref = pd.Timestamp(cfg["ref_date"])

    # ── 1. 로드와 9평 파일 병합 ──
    prev, curr = _scores(D["prev_scores"]), _scores(D["curr_scores"])
    sept_files = D.get("sept_scores", [])
    sept_files = sept_files if isinstance(sept_files, list) else [sept_files]
    merges = []
    if sept_files:
        sept = _scores(sept_files[0])
        for f in sept_files[1:]:
            sept, ch = clean.merge_versions(_scores(f), sept)
            merges.append(ch.assign(파일=Path(f if isinstance(f, str) else f["path"]).name))
        curr = pd.concat([curr[curr["시험명"] != SEPT], sept], ignore_index=True)
    merge_log = pd.concat(merges, ignore_index=True) if merges else pd.DataFrame()
    merge_log.to_csv(out / "sept_merge_log.csv", index=False)
    roster = io.load_roster(D["roster"])
    log(f"[1] 로드: 전년도 {prev['학번'].nunique()}명, 올해 {curr['학번'].nunique()}명, 9평 파일 {len(sept_files)}개 병합 (변경 {len(merge_log)}건)")

    # ── 2. 정제 ──
    curr, manual = clean.apply_manual_fixes(curr, io.load_fixes(D.get("fixes")))
    iss_c, iss_p = clean.find_issues(curr), clean.find_issues(prev)
    curr, prev = clean.apply_auto_fixes(curr, iss_c), clean.apply_auto_fixes(prev, iss_p)
    parts = [f.assign(구분=k) for f, k in ((manual, "올해"), (iss_c, "올해"), (iss_p, "전년도")) if len(f)]
    issues = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["구분", "학번", "시험명", "열", "값", "처리", "수정값", "근거"])
    issues = issues.drop(columns=["row"], errors="ignore")
    issues.to_csv(out / "data_issues.csv", index=False)
    log(f"[2] 정제: {issues['처리'].value_counts().to_dict() if len(issues) else '문제 없음'}")

    # ── 3. 지표와 대상 정의 ──
    prev, curr = metrics.add_metrics(prev), metrics.add_metrics(curr)
    Wp, Wpt, Wc = metrics.wide(prev, "pct3"), metrics.wide(prev, "total"), metrics.wide(curr, "pct3")
    enrolled = roster[(roster["등원일"] <= ref) & (roster["퇴원일"].isna() | (roster["퇴원일"] > ref))]
    level = roster.set_index("학번")["학년"].to_dict()
    cur = set(enrolled["학번"])
    early, private = total_model.has_any(Wc, EARLY), total_model.has_any(Wc, PRIVATE)
    ids_m1 = [i for i in Wc.index if i in cur and early[i]]
    ids_m2 = [i for i in Wc.index if i in cur and not early[i] and private[i]]
    ids_9 = [i for i in Wc.index if i in cur and i not in set(ids_m1) | set(ids_m2) and pd.notna(Wc.at[i, SEPT] if SEPT in Wc else np.nan)]
    log(f"[3] 대상: {GROUP_EXISTING} {len(ids_m1)}명, {GROUP_LATE} {len(ids_m2)}명, {GROUP_SEPT_ONLY} {len(ids_9)}명")

    # ── 4. 총점 모델 ──
    comp = total_model.compare_inputs(Wp)
    comp.to_csv(out / "model_comparison.csv", index=False)
    Wq, eq = total_model.private_equating(Wp, Wc)
    sept_tab, sept_preds = total_model.sept_check(Wp, Wc, Wq, ids_m1, ids_m2)
    sept_tab.to_csv(out / "sept_check.csv", index=False)
    M = {"M1": total_model.fit_total("M1", Wp, Wpt, M1_EXAMS, total_model.has_any(Wp, EARLY)),
         "M2": total_model.fit_total("M2", Wp, Wpt, M2_EXAMS)}
    shifts = ", ".join(f"{e}월 {v['shift']:+.1f}" for e, v in eq.items())
    log(f"[4] 총점 모델: M1 n={M['M1'].n} LOO RMSE {M['M1'].rmse_p:.1f}, M2 n={M['M2'].n} {M['M2'].rmse_p:.1f}; "
        f"사설 실모 연도 차 {shifts}")

    # ── 5. 오차 모델 ──
    TH = {k: error_model.fit_sigma(m.pred_loo, m.y - m.pred_loo) for k, m in M.items()}
    fl = pd.concat([error_model.floor_eval(m.pred_loo, m.y, cap=cap).assign(모델=k) for k, m in M.items()], ignore_index=True)
    fl.to_csv(out / "sigma_floor_eval.csv", index=False)
    cal = error_model.calibration(M["M1"].pred_loo, M["M1"].y, TH["M1"], floor, cap)
    cal.to_csv(out / "sigma_calibration.csv", index=False)
    sig_info = {k: {"theta": [float(x) for x in TH[k]], "floor": floor, "cap": cap,
                    "sigma_at": {str(p): round(float(error_model.sigma(TH[k], p, floor, cap)), 1) for p in (200, 220, 230, 250, 270)}}
                for k in M}
    (out / "sigma_params.json").write_text(json.dumps(sig_info, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[5] 오차 모델: M1 σ(200/220/230/250/270) = {list(sig_info['M1']['sigma_at'].values())}")

    # ── 6. 학생별 수능 예상 ──
    gap = metrics.tam_gap(curr)
    S = []
    for mk, ids, W, exams, grp in (("M1", ids_m1, Wc, M1_EXAMS, GROUP_EXISTING), ("M2", ids_m2, Wq, M2_EXAMS, GROUP_LATE)):
        x = total_model.mean_of(W.loc[ids], exams)
        p, t = M[mk].predict(x)
        sd = error_model.sigma(TH[mk], p, floor, cap)
        for i, xi, pi, ti, si in zip(ids, x, p, t, sd):
            S.append(dict(학번=int(i), 구분=grp, 모델=mk, x=xi, sat_p=pi, sat_t=ti, sd=si,
                          td=M[mk].rmse_t * si / M[mk].rmse_p, gap=float(gap.get(i, 0.0))))
    S = pd.DataFrame(S).set_index("학번")

    # ── 7. 대학 라인 ──
    gcfg = lines.load_groups(cfg["groups"])
    B = lines.usable_units(lines.assign_groups(io.load_placement(D["placement"]), gcfg), gcfg)
    L = lines.line_cuts(B, gcfg)
    pd.DataFrame([{"라인": x["g"], **{f"{k} 컷": x["cut"][k] for k in x["cut"]}, **{f"{k} 학과 수": x["n"][k] for k in x["n"]}} for x in L]).to_csv(out / "line_cuts.csv", index=False)
    st = []
    for grp in (GROUP_EXISTING, GROUP_LATE):
        s = S[S["구분"] == grp]
        if len(s):
            st.append(lines.line_stats(s["sat_p"], s["sd"], L, seed=sim.get("seed", 21)).assign(구분=grp))
    st = pd.concat(st, ignore_index=True) if st else pd.DataFrame()
    if len(st):
        st = st.merge(lines.last_year_shares(Wp[SAT], L)[["라인", "계열", "전년도_비율"]], on=["라인", "계열"], how="left")
    else:
        log("    배치컷을 정할 수 있는 라인이 없습니다. 그룹 설정의 min_units 와 대학명을 확인하세요.")
        st = pd.DataFrame(columns=["라인", "계열", "배치컷", "n", "예상_인원", "하한_5", "상한_95", "안정", "가능", "구분", "전년도_비율"])
    st.to_csv(out / "line_stats.csv", index=False)
    bc = lines.backcheck(M["M1"].pred_loo, M["M1"].y, error_model.sigma(TH["M1"], M["M1"].pred_loo, floor, cap), L)
    bc.to_csv(out / "line_backcheck.csv", index=False)
    log(f"[7] 대학 라인 {len(L)}개, 배치표 {len(B)}개 모집단위")

    # ── 8. 과목 모델과 수능 최저 ──
    D1 = subject_model.features(prev, M1_EXAMS, set(Wp.index[total_model.has_any(Wp, EARLY)]), target=SAT)
    D2 = subject_model.features(prev, M2_EXAMS, set(prev.loc[prev["시험명"].isin(M2_EXAMS), "학번"]), target=SAT)
    D1, D2 = D1[D1["국_g"].notna()], D2[D2["국_g"].notna()]
    T1, T2 = subject_model.train(D1), subject_model.train(D2)
    bt = subject_model.backtest(D1, T1[1], T1[2], n_sim=n_bt, ok=ok, warn=warn)
    bts = bt.groupby("판정").agg(n=("실제", "size"), 평균_예측=("p", "mean"), 실제_충족=("실제", "mean")).reindex(["안정", "경계", "위험"]).reset_index()
    bts.to_csv(out / "minimum_backtest.csv", index=False)
    adj = subject_model.private_adjust(prev, curr)
    F1 = subject_model.features(curr, M1_EXAMS, set(ids_m1))
    F2 = subject_model.features(curr, M2_EXAMS, set(ids_m2), adj=adj)
    RES = {}
    for F, (coef, sg, R) in ((F1, T1), (F2, T2)):
        for sid, r in F.iterrows():
            if any(pd.isna(r[s + "_x"]) for s in subject_model.SUBJ):
                continue
            RES[int(sid)] = subject_model.student_result(sid, r, coef, sg, R, n_sim, ok, warn, compare=True)
    exist = [i for i in ids_m1 if int(i) in RES]
    mt = [{"유형": nm, **{j: sum(subject_model.judge(RES[int(i)]["min"][nm], ok, warn) == j for i in exist) for j in ("안정", "경계", "위험")}}
          for nm, _, _ in subject_model.PATTERNS]
    pd.DataFrame(mt).to_csv(out / "minimum_table.csv", index=False)
    bott = [RES[int(i)] for i in exist if RES[int(i)]["bott"]]
    cmp_rows = []
    for name, get in (("한 등급 하락 시 충족 확률 감소폭", lambda r: r["bott_alt"]["한 등급 하락"]),
                      ("과목별 1σ 하락 시 감소폭", lambda r: r["bott_alt"]["1σ 하락"]),
                      ("실패 시뮬레이션의 원인 과목 (채택)", lambda r: r["bott"])):
        vc = pd.Series([get(r) for r in bott]).value_counts()
        cmp_rows.append({"정의": name, **{k: int(vc.get(k, 0)) for k in ("국어", "수학", "영어", "탐구")}})
    pd.DataFrame(cmp_rows).to_csv(out / "bottleneck_compare.csv", index=False)
    log(f"[8] 과목 모델: 학습 {len(D1)}명/{len(D2)}명, 최저 판정 {len(RES)}명, 경계 학생 {len(bott)}명")

    # ── 9. 대시보드 ──
    rows = {sid: g for sid, g in curr.groupby("학번")}
    students = []
    for sid in ids_m1 + ids_m2 + ids_9:
        i = int(sid)
        g = rows[sid]
        mon = {e: (None if (e not in Wc.columns or pd.isna(Wc.at[sid, e])) else round(float(Wc.at[sid, e]), 1)) for e in PRE_SEPT}
        act = Wc.at[sid, SEPT] if SEPT in Wc.columns else np.nan
        o = {"id": i, "g": level.get(sid, ""), "mon": mon, "chg": _subject_change(g), "gr": _grades(g)}
        if sid in S.index:
            r = S.loc[sid]
            # 대시보드의 9평 예측은 9평 전에 고정한 예측입니다 (기존 인원은 보정한 7·8월까지 포함)
            key = "3~8월 보정" if r["모델"] == "M1" else "보정 7·8월 (6평 이후 등원)"
            pr, rr = sept_preds[key]
            o["grp"] = r["구분"]
            o["n9"] = {"pred": None if sid not in pr.index or pd.isna(pr[sid]) else round(float(pr[sid]), 1),
                       "act": None if pd.isna(act) else round(float(act), 1), "r": round(rr, 1)}
            o["sat"] = {"p": round(float(r["sat_p"]), 1), "t": round(float(r["sat_t"]), 1),
                        "rp": round(M[r["모델"]].rmse_p, 1), "rt": round(M[r["모델"]].rmse_t, 1),
                        "sd": round(float(r["sd"]), 1), "gap": round(float(r["gap"]), 1), "td": round(float(r["td"]), 1)}
            if i in RES:
                o.update({k: v for k, v in RES[i].items() if k != "bott_alt"})
        else:
            o["grp"] = GROUP_SEPT_ONLY
            o["n9"] = {"pred": None, "act": None if pd.isna(act) else round(float(act), 1), "r": None}
        students.append(o)
    meta = {"patterns": [p[0] for p in subject_model.PATTERNS], "date": cfg["ref_date"],
            "lines": [{"g": x["g"], "cut": x["cut"]} for x in L], "units": lines.units_for_dashboard(B)}
    payload = {"meta": meta, "students": students}
    (out / "dashboard_data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    dc = cfg["dashboard"]
    page = dashboard.render(payload, dc["template"], out / "dashboard.html", dc.get("title", "수능 예측 대시보드"),
                            dc.get("eyebrow", ""), dc.get("subtitle", ""), dc.get("font_dir"))
    log(f"[9] 대시보드: {page} ({len(students)}명)")

    # ── 10. 학생별 결과표와 요약 ──
    sv = S.copy()
    sv["3합6"] = [RES.get(int(i), {}).get("min", {}).get("3합6") for i in sv.index]
    sv["병목"] = [RES.get(int(i), {}).get("bott", "") for i in sv.index]
    sv.round(1).to_csv(out / "students.csv")
    summary = [
        "# 실행 요약", "",
        f"기준일 {cfg['ref_date']}. 대상 {GROUP_EXISTING} {len(ids_m1)}명, {GROUP_LATE} {len(ids_m2)}명, {GROUP_SEPT_ONLY} {len(ids_9)}명.", "",
        "## 입력 조합별 LOO RMSE (전년도 수능)", "", _md_table(comp), "",
        "## 9평 사전 예측 점검 (치우침 = 실제 − 예측)", "", _md_table(sept_tab, 2), "",
        "## 오차 모델 비교 (LOO)", "", _md_table(fl, 4), "",
        "## 점수대별 σ", "", _md_table(pd.DataFrame([{"모델": k, **v["sigma_at"]} for k, v in sig_info.items()])), "",
        "## 대학 라인별 전망", "", _md_table(st[st["구분"] == GROUP_EXISTING].drop(columns=["구분"]), 2), "",
        "## 전년도 재현 점검 (자연 컷)", "", _md_table(bc), "",
        "## 수능 최저 백테스트 (전년도, LOO)", "", _md_table(bts, 3), "",
        f"## 수능 최저 판정 ({GROUP_EXISTING})", "", _md_table(pd.DataFrame(mt)), "",
        "## 병목 과목 정의 비교", "", _md_table(pd.DataFrame(cmp_rows)), "",
        "## 데이터 정제", "", _md_table(issues[["구분", "학번", "시험명", "열", "값", "처리", "수정값", "근거"]]) if len(issues) else "문제 없음", "",
    ]
    (out / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    log(f"[10] 결과 파일: {out}")
    return {"students": students, "lines": L, "line_stats": st, "issues": issues, "sept_check": sept_tab,
            "model_comparison": comp, "floor_eval": fl, "backtest": bts, "bottleneck": pd.DataFrame(cmp_rows)}
