"""예측 고정 기록.

실제 예측 파일은 저장소에 올리지 않고 SHA-256 해시만 preregistration/hashes.csv 에 남깁니다.
수능이 끝난 뒤 같은 파일의 해시가 기록과 같으면, 그 파일이 기록한 날짜 이전에 이미 있었다는 증거가 됩니다.
파일을 한 글자라도 바꾸면 해시가 달라지므로 원본은 그대로 보관합니다. 엑셀에서 열었다가 저장해도 바뀝니다.

사용법
  python scripts/freeze_predictions.py 예측파일.csv --label "9평 반영 수능 예측"   # 기록 추가
  python scripts/freeze_predictions.py 예측파일.csv --verify                         # 기록과 대조
"""
import argparse
import csv
import datetime as dt
import hashlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "preregistration" / "hashes.csv"
FIELDS = ["기록일", "설명", "파일명", "sha256"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--label", default="")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    path = Path(a.file).resolve()
    if REPO in path.parents and "preregistration" not in path.parts:
        print("주의: 예측 파일이 저장소 안에 있습니다. 저장소 밖으로 옮긴 뒤 실행하세요.")
        return
    digest = sha256(path)
    rows = list(csv.DictReader(LOG.open(encoding="utf-8"))) if LOG.exists() else []
    if a.verify:
        hit = [r for r in rows if r["sha256"] == digest]
        print(f"{path.name}: {digest}")
        print(f"일치: {hit[0]['기록일']} {hit[0]['설명']}" if hit else "일치하는 기록이 없습니다.")
        return
    new = not LOG.exists()
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"기록일": dt.date.today().isoformat(), "설명": a.label, "파일명": path.name, "sha256": digest})
    print(f"기록함: {path.name} {digest}")


if __name__ == "__main__":
    main()
