"""파이프라인 실행.

사용법
  python scripts/run_pipeline.py                          # 가상 데이터(config/sample.json)
  python scripts/run_pipeline.py --config config/real.private.json --out outputs/real
  python scripts/run_pipeline.py --n-sim 1000             # 빠르게 확인할 때
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from suneung.pipeline import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(REPO / "config/sample.json"))
    ap.add_argument("--out", default=str(REPO / "outputs"))
    ap.add_argument("--n-sim", type=int, default=None, help="학생별 시뮬레이션 횟수 (기본: 설정 파일 값)")
    a = ap.parse_args()
    run(a.config, a.out, n_sim=a.n_sim)


if __name__ == "__main__":
    main()
