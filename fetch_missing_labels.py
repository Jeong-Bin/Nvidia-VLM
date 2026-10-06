#!/usr/bin/env python3
"""로컬에 mp4 만 있고 라벨이 없는 클립의 타임스탬프/egomotion 을 채운다.

왜 필요한가:
  pav_sample 의 라벨은 download_more_clips.py 로 받은 청크 0~19 것뿐이다.
  라벨 셋(test_label_333_diff.json)을 늘리면서 NAS 전체에서 골라 온 클립은
  mp4 만 복사돼, egomotion 과 timestamps.parquet 가 없다. 그러면
  --use-egomotion 으로 돌려도 그 클립에는 자차 행동 사실이 들어가지 않는다
  (실측 20261002_175140_eval: 333 중 102 클립, 청크 47~3113 에 흩어짐).
  그 102 개에 드문 카테고리가 몰려 있어(Unpaved 19 중 18) 영향이 크다.

어디서 받는가:
  timestamps.parquet  NAS 카메라 청크 zip 안에 mp4 와 같이 들어 있다.
  egomotion           NAS 에 없다 - HuggingFace 의 labels/egomotion 청크
                      zip(약 38MB)을 받아 필요한 클립만 꺼낸다.

청크 하나에 대상 클립이 1~2 개뿐이라 zip 을 통째로 풀지 않고 필요한
파일만 꺼낸다. 받은 zip 은 꺼낸 뒤 지운다(--keep-zip 이면 남긴다).
이미 있는 파일은 건너뛰므로 중간에 끊겨도 다시 돌리면 이어진다.

Usage:
  python fetch_missing_labels.py                      # 라벨 파일의 누락 클립 전부
  python fetch_missing_labels.py --dry-run            # 무엇을 받을지만 보여준다
  python fetch_missing_labels.py --uuids uuids.txt    # 이 목록만
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd

import config
from clip_source import NAS_CAMERA_DIR

ROOT = Path(__file__).resolve().parent
REPO = "nvidia/PhysicalAI-Autonomous-Vehicles"
VIEW = "camera_front_wide_120fov"
CAMERA_DIR = ROOT / "pav_sample" / "camera" / VIEW
EGO_DIR = ROOT / "pav_sample" / "labels" / "egomotion"
CLIP_INDEX = Path(NAS_CAMERA_DIR).parent / "clip_index.parquet"


def ts_name(uuid: str) -> str:
    return f"{uuid}.{VIEW}.timestamps.parquet"


def ego_name(uuid: str) -> str:
    return f"{uuid}.egomotion.parquet"


def has_ego(uuid: str) -> bool:
    """풀린 parquet 이든 로컬 청크 zip 안이든 egomotion 이 있는가."""
    import egomotion as EM
    return EM.has_egomotion(uuid)


def has_ts(uuid: str) -> bool:
    return (CAMERA_DIR / ts_name(uuid)).exists()


def extract_member(zp: Path, member: str, dest: Path) -> bool:
    """zip 에서 파일 하나를 dest 로 꺼낸다. 없으면 False.

    임시 파일에 쓴 뒤 rename 한다 - 중간에 끊기면 반쪽짜리 parquet 이
    남아, 다음 실행이 "이미 있다"며 건너뛰고 읽을 때 깨진다.
    """
    with zipfile.ZipFile(zp) as z:
        try:
            data = z.read(member)
        except KeyError:
            return False
    tmp = dest.with_name(dest.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, dest)
    return True


def target_uuids(args) -> list[str]:
    if args.uuids:
        return [l.strip() for l in open(args.uuids, encoding="utf-8")
                if l.strip()]
    data = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    clips = data.get("clips", data)
    # mp4 가 로컬에 있는 클립만 - 영상이 없으면 라벨을 채워도 추론할 수 없다.
    return [u for u in clips if (CAMERA_DIR / f"{u}.{VIEW}.mp4").exists()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "test_label_333_diff.json"),
                    help="이 라벨 파일의 클립 중 누락된 것을 채운다")
    ap.add_argument("--uuids", default=None,
                    help="라벨 파일 대신 이 uuid 목록(한 줄에 하나)만")
    ap.add_argument("--cache-dir", default=str(ROOT / ".hf_download_cache"))
    ap.add_argument("--keep-zip", action="store_true",
                    help="꺼낸 뒤에도 egomotion 청크 zip 을 남긴다")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    uuids = target_uuids(args)
    need_ego = [u for u in uuids if not has_ego(u)]
    need_ts = [u for u in uuids if not has_ts(u)]
    todo = sorted(set(need_ego) | set(need_ts))
    print(f"[info] 대상 {len(uuids)} 클립 중 egomotion 없음 {len(need_ego)}, "
          f"timestamps 없음 {len(need_ts)}")
    if not todo:
        print("[info] 채울 것이 없습니다")
        return

    ci = pd.read_parquet(CLIP_INDEX)
    chunk_of = ci["chunk"].to_dict()
    unknown = [u for u in todo if u not in chunk_of]
    if unknown:
        print(f"[warn] clip_index 에 없는 uuid {len(unknown)}개는 건너뜀: "
              f"{unknown[:3]}")
    by_chunk = defaultdict(list)
    for u in todo:
        if u in chunk_of:
            by_chunk[int(chunk_of[u])].append(u)
    print(f"[info] 청크 {len(by_chunk)}개에 흩어져 있음 "
          f"(egomotion zip 약 {len([c for c, us in by_chunk.items() if any(u in need_ego for u in us)]) * 38}MB 다운로드)")
    if args.dry_run:
        for c in sorted(by_chunk)[:10]:
            print(f"  chunk_{c:04d}: {by_chunk[c]}")
        print("[dry-run] 끝")
        return

    from huggingface_hub import hf_hub_download

    CAMERA_DIR.mkdir(parents=True, exist_ok=True)
    EGO_DIR.mkdir(parents=True, exist_ok=True)
    got_ts = got_ego = 0
    failed = []
    chunks = sorted(by_chunk)
    for i, c in enumerate(chunks, 1):
        us = by_chunk[c]
        # 1) timestamps - NAS 카메라 zip
        for u in us:
            if u in need_ts and not has_ts(u):
                zp = Path(NAS_CAMERA_DIR) / VIEW / f"{VIEW}.chunk_{c:04d}.zip"
                try:
                    if extract_member(zp, ts_name(u), CAMERA_DIR / ts_name(u)):
                        got_ts += 1
                    else:
                        failed.append((u, "timestamps: zip 안에 없음"))
                except (OSError, zipfile.BadZipFile) as e:
                    failed.append((u, f"timestamps: {type(e).__name__}: {e}"))
        # 2) egomotion - HuggingFace
        want = [u for u in us if u in need_ego and not has_ego(u)]
        if want:
            rp = f"labels/egomotion/egomotion.chunk_{c:04d}.zip"
            try:
                zp = hf_hub_download(REPO, rp, repo_type="dataset",
                                     cache_dir=args.cache_dir)
                for u in want:
                    if extract_member(Path(zp), ego_name(u), EGO_DIR / ego_name(u)):
                        got_ego += 1
                    else:
                        failed.append((u, f"egomotion: {rp} 안에 없음"))
                if not args.keep_zip:
                    # HF 캐시의 blob 실체를 지운다 (symlink 경유)
                    for p in {os.path.realpath(zp), zp}:
                        try:
                            os.remove(p)
                        except OSError:
                            pass
            except Exception as e:
                for u in want:
                    failed.append((u, f"egomotion: {type(e).__name__}: {e}"))
        print(f"[{i}/{len(chunks)}] chunk_{c:04d}  timestamps {got_ts}  "
              f"egomotion {got_ego}", flush=True)

    # egomotion 인덱스는 lru_cache 라 새 파일을 보려면 비운다.
    import egomotion as EM
    EM._zip_index.cache_clear()
    still_ego = [u for u in uuids if not EM.has_egomotion(u)]
    still_ts = [u for u in uuids if not has_ts(u)]
    print(f"\n[done] 받은 파일: timestamps {got_ts}, egomotion {got_ego}")
    print(f"[done] 여전히 없음: egomotion {len(still_ego)}, "
          f"timestamps {len(still_ts)}")
    for u, why in failed[:20]:
        print(f"  !! {u}  {why}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
