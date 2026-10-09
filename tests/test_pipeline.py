"""가상 데이터로 전체 파이프라인을 돌려 보는 통합 테스트."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from suneung import clean, io
from suneung.pipeline import load_config, run

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    data = tmp / "data"
    subprocess.run([sys.executable, str(REPO / "scripts/make_sample_data.py"), "--out", str(data), "--seed", "2"], check=True)
    # 엑셀 복붙 사고처럼 같은 행을 한 번 더 넣어 둡니다 (결과에는 영향이 없어야 함)
    for f in ("scores_curr.csv", "sept_v2.csv"):
        d = pd.read_csv(data / f)
        pd.concat([d, d.iloc[[1]]]).to_csv(data / f, index=False)
    cfg = load_config(REPO / "config/sample.json")
    for k, v in list(cfg["data"].items()):
        cfg["data"][k] = [str(data / Path(x).name) for x in v] if isinstance(v, list) else str(data / Path(v).name)
    out = tmp / "out"
    return data, out, run(cfg, out, n_sim=800, log=lambda *a: None)


def test_outputs_and_dashboard(result):
    data, out, res = result
    page = (out / "dashboard.html").read_text(encoding="utf-8")
    assert "__DATA__" not in page and "__TITLE__" not in page
    payload = json.loads((out / "dashboard_data.json").read_text(encoding="utf-8"))
    assert payload["meta"]["judge"] == {"ok": 0.8, "warn": 0.4}
    groups = {s["grp"] for s in payload["students"]}
    assert groups == {"기존 인원", "6평 이후 등원", "9평만"}
    assert all("sat" in s for s in payload["students"] if s["grp"] != "9평만")
    gd = [s["gd"] for s in payload["students"] if "gd" in s]
    assert gd and all(len(v) == 9 and abs(sum(v) - 1) < 0.01 for g in gd for v in g.values())
    assert len(res["excluded"]) == 1                       # 재원 중이지만 4과목 완비 시험이 없는 학생


def test_duplicates_are_logged(result):
    _, _, res = result
    assert (res["issues"]["처리"] == "중복 합침").sum() == 2


def test_injected_errors_handled(result):
    data, _, res = result
    inj, iss = pd.read_csv(data / "injected_errors.csv"), res["issues"]
    expect = {"보정": "보정", "확인 필요": clean.UNRESOLVED, "확인 필요 → 수동 보정": "수동 보정"}
    for _, e in inj.iterrows():
        hit = iss[(iss["학번"] == e["학번"]) & (iss["시험명"].astype(str) == str(e["시험명"])) & (iss["열"] == e["열"])]
        assert len(hit) == 1, e.to_dict()
        assert hit.iloc[0]["처리"] == expect[e["예상처리"]]


def test_all_injected_errors_found_without_manual_fixes(result):
    """수동 보정 목록 없이도 자동 검출만으로 다섯 건을 모두 찾는지 확인합니다."""
    data, _, _ = result
    curr, _ = clean.dedupe(io.load_scores(data / "scores_curr.csv"))
    v1, _ = clean.dedupe(io.load_scores(data / "sept_v1.csv"))
    v2, _ = clean.dedupe(io.load_scores(data / "sept_v2.csv"))
    sept, _ = clean.merge_versions(v2, v1)
    iss = clean.find_issues(pd.concat([curr, sept], ignore_index=True))
    inj = pd.read_csv(data / "injected_errors.csv")
    for _, e in inj.iterrows():
        hit = iss[(iss["학번"] == e["학번"]) & (iss["시험명"].astype(str) == str(e["시험명"])) & (iss["열"] == e["열"])]
        assert len(hit) == 1, e.to_dict()
        assert hit.iloc[0]["처리"] == ("보정" if e["예상처리"] == "보정" else "확인 필요")
    assert len(iss) == len(inj)                            # 오탐 없음


def test_readme_numbers_are_reproduced(result):
    """README '가상 데이터에서 재현되는 경향' 표의 숫자를 고정합니다 (시뮬레이션 횟수와 무관한 값만)."""
    _, _, res = result
    sc = res["sept_check"].set_index("방식")
    assert sc.at["3~8월 원점수", "치우침"] == pytest.approx(-7.43, abs=0.05)
    assert sc.at["3~8월 보정", "치우침"] == pytest.approx(0.10, abs=0.05)
    fl = res["floor_eval"].query("모델 == 'M1'").set_index("방식")["Brier"]
    assert fl["단일 σ"] == pytest.approx(0.0781, abs=0.0005)
    assert 0.0655 <= fl[["floor 0", "floor 8", "floor 10", "floor 12", "floor 14"]].min()
    assert fl[["floor 0", "floor 8", "floor 10", "floor 12", "floor 14"]].max() <= 0.0695
    comp = res["model_comparison"].set_index("입력")
    assert comp.loc["3~6월 + 9평", "LOO_RMSE"].iloc[0] == pytest.approx(24.69, abs=0.01)
