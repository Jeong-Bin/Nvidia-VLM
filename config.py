#!/usr/bin/env python3
"""클립 파이프라인이 공유하는 경로 설정 - 단일 진실 공급원.

왜 이 파일이 필요한가:
  클래스 정의와 정답 라벨의 기본값이 예전에는 다섯 군데에 흩어져 있었다
  (edge_case_mining.py, evaluate_labels.py, aggregate_clip.py, run_eval.sh,
  run_video_C.sh). 그 값들이 서로 어긋나면서 실제로 사고가 났다 - 같은
  실행 결과를 aggregate_clip.py 는 D 기준으로 집계하고 evaluate_labels.py
  는 C 기준으로 검증한 적이 있다. 숫자만 보면 구분이 안 되는 종류의 오류다.

  더 고약했던 것은 셸 스크립트가 자기 기본값을 항상 명령행으로 넘겨서,
  파이썬 쪽 default 를 아무리 고쳐도 반영되지 않았다는 점이다. 그래서
  기본값은 여기 한 곳에만 두고, 셸은 사용자가 명시했을 때만 플래그를
  넘긴다 (미지정이면 플래그 자체를 생략 -> 아래 값이 실제로 쓰인다).

바꾸는 법:
  이 파일의 두 줄만 고치면 추론/집계/채점/시각화가 모두 따라온다.
  일회성으로 다른 파일을 쓰려면 --scene-json / --labels 로 넘기면 된다.

셸에서 읽는 법:
  python3 -c 'import config; print(config.SCENE_JSON)'
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# 클래스 분류 체계. 모델 프롬프트의 클래스 메뉴, 집계 대상 목록,
# 채점 시 라벨 이름 검증이 모두 이 파일을 본다.
SCENE_JSON = ROOT / "scene_category_3.0.json"

# 사람이 만든 정답 라벨. 채점(evaluate_labels.py --labels)과 시각화
# 패널의 GT 표기(--gt-labels)가 함께 쓴다.
LABELS_JSON = ROOT / "test_label_333_diff2.json"


if __name__ == "__main__":
    # 셸 스크립트가 값을 물어볼 때 쓴다. 인자 없으면 둘 다 출력.
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else None
    if name == "scene":
        print(SCENE_JSON)
    elif name == "labels":
        print(LABELS_JSON)
    else:
        print(f"SCENE_JSON  = {SCENE_JSON}")
        print(f"LABELS_JSON = {LABELS_JSON}")

# 이름을 class 로 바꾸기 전에 끝난 실행의 CSV 와 run_config.json 은 옛 이름을
# 그대로 갖고 있다. 결과 파일은 고치지 않고 읽을 때 새 이름으로 맞춘다.
OLD_NAMES = {"categories": "classes", "n_categories": "n_classes", "category": "class",
             "score_categories": "score_classes", "unknown_categories": "unknown_classes",
             "viz_per_category": "viz_per_class"}


def new_names(d: dict) -> dict:
    return {OLD_NAMES.get(k, k): v for k, v in d.items()}


def read_run_config(path) -> dict:
    """run_config.json 을 읽어 key 와 argv 의 옛 이름을 맞춘다."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    for block in ("key", "argv"):
        if block in cfg:
            cfg[block] = new_names(cfg[block])
    return cfg

