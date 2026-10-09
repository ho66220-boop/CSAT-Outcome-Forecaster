"""가상 데이터로 전체 파이프라인을 돌려 보는 통합 테스트."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from suneung.pipeline import load_config, run

REPO = Path(__file__).resolve().parents[1]


def test_end_to_end(tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, str(REPO / "scripts/make_sample_data.py"), "--out", str(data), "--seed", "2"], check=True)
    cfg = load_config(REPO / "config/sample.json")
    for k, v in list(cfg["data"].items()):
        cfg["data"][k] = [str(data / Path(x).name) for x in v] if isinstance(v, list) else str(data / Path(v).name)
    out = tmp_path / "out"
    res = run(cfg, out, n_sim=800, log=lambda *a: None)

    page = (out / "dashboard.html").read_text(encoding="utf-8")
    assert "__DATA__" not in page and "__TITLE__" not in page
    payload = json.loads((out / "dashboard_data.json").read_text(encoding="utf-8"))
    groups = {s["grp"] for s in payload["students"]}
    assert groups == {"기존 인원", "6평 이후 등원", "9평만"}
    assert all("sat" in s for s in payload["students"] if s["grp"] != "9평만")

    # 일부러 넣은 오류를 모두 찾았는지 확인
    inj = pd.read_csv(data / "injected_errors.csv")
    iss = res["issues"]
    for _, e in inj.iterrows():
        hit = iss[(iss["학번"] == e["학번"]) & (iss["시험명"].astype(str) == str(e["시험명"])) & (iss["열"] == e["열"])]
        assert len(hit) == 1, e.to_dict()
        assert hit.iloc[0]["처리"] in e["예상처리"]

    # 사설 실모를 그대로 쓰면 9평을 높게 예측하고, 보정하면 치우침이 줄어야 함
    sc = res["sept_check"].set_index("방식")
    assert sc.at["3~8월 원점수", "치우침"] < -3
    assert abs(sc.at["3~8월 보정", "치우침"]) < abs(sc.at["3~8월 원점수", "치우침"])
