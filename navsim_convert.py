#!/usr/bin/env python3
"""NavSim(OpenScene) test 스플릿을 파이프라인이 읽는 청크 zip 으로 바꾼다.

원본은 로그별 폴더의 2Hz JPG 프레임이고, 파일 이름이 해시라 순서는 메타데이터로만
알 수 있다. 파이프라인은 클립별 mp4 가 무압축 zip 에 든 형식을 읽는다.

클립 단위는 메타데이터의 scene_token 이다(거의 20초). NAVSIM 평가의 "scene"
(14프레임, 7초, 서로 겹침)과는 다른 단위다. 임의로 자르지 않는다.

  raw/*.tgz --(고른 카메라만 추출)--> <work>/frames --(H.264)--> <work>/mp4
            --(무압축 zip)--> <out>/camera/<카메라>/<카메라>.chunk_NNNN.zip
  카메라는 --cameras 로 고른다(기본 CAM_F0, all 이면 8대 전부). 카메라가 달라도
  클립 구성과 청크 번호는 같아야 한다 - 청크 단위로 라벨링을 나누기 때문이다.
  <out>/clips.json 클립 -> 로그, 프레임 token, 시각. navtest token 으로 클립을
                   고를 때 쓴다.
  <out>/ego_2hz.json 클립 -> 프레임별 자차 속도, 가속도, 방향. egomotion.py 가
                   읽는다. 이것만 다시 만들 때는 --ego-only.

H.264 로 만든다 - NVIDIA 원본의 HEVC 와 달리 어느 브라우저에서나 재생된다.

중간에 끊겨도 다시 실행하면 끝난 샤드와 만들어 둔 mp4 는 건너뛴다.
공개 pkl 은 읽는 순간 코드가 돌 수 있어 numpy 복원만 허용해서 읽는다.

  python3 navsim_convert.py --raw /mnt/nas/NAVSIM/20261105/raw \\
      --out /mnt/nas/NAVSIM/20261105 --work /tmp/navsim_work
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import tarfile
import zipfile
from pathlib import Path

import av
from PIL import Image

ALL_CAMERAS = ["CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0"]


def view_of(cam: str) -> str:
    return cam.lower()
FPS = 2                 # OpenScene 은 2Hz 다
MIN_FRAMES = 36         # 18초. 이보다 짧은 scene 은 대부분 로그 끝의 조각이다
CHUNK_CLIPS = 100       # 청크 하나의 클립 수. 라벨링 분배의 단위가 된다
SHARD_RE = re.compile(r"openscene_sensor_test_camera_(\d+)\.tgz$")
PICKLE_ALLOW = {("numpy.core.multiarray", "_reconstruct"),
                ("numpy._core.multiarray", "_reconstruct"),
                ("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy.core.multiarray", "scalar"),
                ("numpy._core.multiarray", "scalar")}


class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in PICKLE_ALLOW:
            raise pickle.UnpicklingError(f"허용하지 않은 타입: {module}.{name}")
        return getattr(__import__(module, fromlist=[name]), name)


def extract_metadata(raw: Path, work: Path) -> Path:
    """메타데이터 pkl 을 푼다. 이름은 파일명만 써서 tar 경로를 따라가지 않는다."""
    meta = work / "meta"
    if (meta / ".done").exists():
        return meta
    meta.mkdir(parents=True, exist_ok=True)
    with tarfile.open(raw / "openscene_metadata_test.tgz", "r|gz") as tar:
        for m in tar:
            if m.isfile() and m.name.endswith(".pkl"):
                (meta / Path(m.name).name).write_bytes(tar.extractfile(m).read())
    (meta / ".done").touch()
    return meta


def ego_sample(x: dict) -> tuple:
    """프레임 메타 -> (vx, vy, ax, yaw_deg). 속도와 가속도는 차량 좌표계다.

    ego2global_rotation 은 (w, x, y, z) 순서다. 함께 든 ego2global 행렬의
    방향과 전 프레임에서 일치함을 확인했다."""
    vx, vy, ax, _ = (float(v) for v in x["ego_dynamic_state"])
    w, qx, qy, qz = (float(v) for v in x["ego2global_rotation"])
    yaw = math.degrees(math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
    return vx, vy, ax, yaw


def write_ego(keep: dict, clips: list, out: Path) -> None:
    """클립별 자차 시계열을 파일 하나로 쓴다. 전체가 수 MB 라 나누지 않는다."""
    doc = {"_meta": {"source": "OpenScene v1.1 test ego_dynamic_state", "hz": FPS,
                     "clips": len(clips)},
           "clips": {}}
    for t in clips:
        fr = keep[t]["frames"]
        doc["clips"][t] = {"t_us": [f[0] for f in fr],
                           **{k: [round(f[3][i], 4) for f in fr]
                              for i, k in enumerate(("vx", "vy", "ax", "yaw_deg"))}}
    tmp = out / "ego_2hz.json.tmp"
    tmp.write_text(json.dumps(doc), encoding="utf-8")
    os.replace(tmp, out / "ego_2hz.json")
    print(f"[자차] ego_2hz.json: 클립 {len(clips)}개", flush=True)


def load_scenes(meta: Path) -> dict:
    """scene_token -> 시간순 프레임 정보."""
    scenes: dict[str, dict] = {}
    for p in sorted(meta.glob("*.pkl")):
        with open(p, "rb") as f:
            frames = SafeUnpickler(f).load()
        for x in frames:
            s = scenes.setdefault(x["scene_token"], {
                "log": p.stem, "scene_name": x["scene_name"],
                "map_location": x["map_location"], "frames": []})
            s["frames"].append((x["timestamp"], x["token"],
                                {c: v["data_path"] for c, v in x["cams"].items()},
                                ego_sample(x)))
    for s in scenes.values():
        s["frames"].sort()
    return scenes


def shard_mark(work: Path, no: int, cam: str) -> Path:
    # CAM_F0 표식은 카메라 선택 기능 전의 이름을 그대로 쓴다.
    return work / (f".shard_{no}.done" if cam == "CAM_F0" else f".shard_{no}.{cam}.done")


def extract_frames(raw: Path, work: Path, needed: dict, shards) -> None:
    """샤드에서 필요한 프레임만 꺼낸다. 메타데이터가 준 경로와 맞는 것만 쓴다.

    needed: 카메라 -> 경로 집합. 샤드는 한 번만 읽고 고른 카메라를 함께 꺼낸다."""
    out = work / "frames"
    files = sorted(raw.glob("openscene_sensor_test_camera_*.tgz"),
                   key=lambda p: int(SHARD_RE.search(p.name).group(1)))
    for p in files:
        no = int(SHARD_RE.search(p.name).group(1))
        if shards is not None and no not in shards:
            continue
        todo = [c for c in needed if not shard_mark(work, no, c).exists()]
        if not todo:
            continue
        want = set().union(*(needed[c] for c in todo))
        n = 0
        with tarfile.open(p, "r|gz") as tar:
            for m in tar:
                key = m.name.split("sensor_blobs/test/", 1)[-1]
                if m.isfile() and key in want:
                    dst = out / key
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    dst.write_bytes(tar.extractfile(m).read())
                    n += 1
        for c in todo:
            shard_mark(work, no, c).touch()
        print(f"[추출] 샤드 {no}: {n}장 ({', '.join(todo)})", flush=True)


def encode(task) -> tuple[str, str | None]:
    """프레임 목록을 mp4 하나로. 실패하면 (token, 사유)."""
    token, paths, dst = task
    dst = Path(dst)
    if dst.exists():
        return token, None
    tmp = dst.with_suffix(".tmp.mp4")
    try:
        with av.open(str(tmp), "w", format="mp4",
                     options={"movflags": "+faststart"}) as c:
            s = None
            for path in paths:
                with Image.open(path) as im:
                    frame = av.VideoFrame.from_image(im.convert("RGB"))
                if s is None:
                    s = c.add_stream("libx264", rate=FPS)
                    s.width, s.height, s.pix_fmt = frame.width, frame.height, "yuv420p"
                    # 키프레임을 촘촘히 둔다. 2fps 라 라벨러가 앞뒤로 옮겨 볼 때
                    # 키프레임 간격이 곧 탐색 지연이다.
                    s.options = {"crf": "20", "preset": "medium", "g": "4", "threads": "2"}
                for pkt in s.encode(frame):
                    c.mux(pkt)
            for pkt in s.encode():
                c.mux(pkt)
        os.replace(tmp, dst)
        return token, None
    except Exception as e:                       # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return token, f"{type(e).__name__}: {e}"


def write_chunks(clips: list, mp4_dir: Path, out: Path, VIEW: str) -> dict:
    """mp4 를 무압축 zip 으로 묶는다. 파이프라인은 zip 안을 바이트 구간으로 읽는다."""
    d = out / "camera" / VIEW
    d.mkdir(parents=True, exist_ok=True)
    where = {}
    for no, i in enumerate(range(0, len(clips), CHUNK_CLIPS)):
        name = f"{VIEW}.chunk_{no:04d}.zip"
        tmp = d / (name + ".tmp")
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for token in clips[i:i + CHUNK_CLIPS]:
                z.write(mp4_dir / f"{token}.{VIEW}.mp4", f"{token}.{VIEW}.mp4")
                where[token] = no
        os.replace(tmp, d / name)
        print(f"[청크] {name}: {len(clips[i:i + CHUNK_CLIPS])}개", flush=True)
    return where


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, type=Path, help="tgz 가 있는 폴더")
    ap.add_argument("--out", required=True, type=Path, help="camera/ 와 clips.json 을 둘 폴더")
    ap.add_argument("--work", required=True, type=Path, help="로컬 작업 폴더 (약 30GB)")
    ap.add_argument("--cameras", default="CAM_F0",
                    help="쉼표로 구분(예: CAM_L0,CAM_R0) 또는 all (기본 CAM_F0)")
    ap.add_argument("--min-frames", type=int, default=MIN_FRAMES)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--shards", default=None,
                    help="시험용. 이 샤드만 쓴다(예: 0,3). 프레임이 다 모이지 않은 "
                         "클립은 건너뛴다")
    ap.add_argument("--limit-clips", type=int, default=None, help="시험용")
    ap.add_argument("--ego-only", action="store_true",
                    help="clips.json 의 클립으로 ego_2hz.json 만 다시 만든다")
    args = ap.parse_args()
    shards = {int(x) for x in args.shards.split(",")} if args.shards else None

    cams = ALL_CAMERAS if args.cameras == "all" else [c.strip().upper() for c in args.cameras.split(",")]
    bad = [c for c in cams if c not in ALL_CAMERAS]
    if bad:
        raise SystemExit(f"알 수 없는 카메라: {bad} (가능: {ALL_CAMERAS})")

    args.work.mkdir(parents=True, exist_ok=True)
    scenes = load_scenes(extract_metadata(args.raw, args.work))
    keep = {t: s for t, s in scenes.items() if len(s["frames"]) >= args.min_frames}
    print(f"[메타] scene {len(scenes)}개, {args.min_frames}장 이상 {len(keep)}개, "
          f"카메라 {', '.join(cams)}", flush=True)

    if args.ego_only:
        doc = json.loads((args.out / "clips.json").read_text(encoding="utf-8"))
        write_ego(keep, list(doc["clips"]), args.out)
        return

    needed = {c: {f[2][c] for s in keep.values() for f in s["frames"]} for c in cams}
    extract_frames(args.raw, args.work, needed, shards)

    frames = args.work / "frames"
    order = sorted(keep, key=lambda t: (keep[t]["log"], keep[t]["frames"][0][0]))
    cj = args.out / "clips.json"
    old = json.loads(cj.read_text(encoding="utf-8")) if cj.exists() else None
    mp4_dir = args.work / "mp4"
    mp4_dir.mkdir(exist_ok=True)
    summary = {}
    for cam in cams:
        VIEW = view_of(cam)
        ready, incomplete = [], []
        for t in order:
            (ready if all((frames / f[2][cam]).exists() for f in keep[t]["frames"])
             else incomplete).append(t)
        if incomplete:
            print(f"[경고] {cam}: 프레임이 모자란 클립 {len(incomplete)}개 건너뜀", flush=True)
        if args.limit_clips:
            ready = ready[:args.limit_clips]
        tasks = [(t, [str(frames / f[2][cam]) for f in keep[t]["frames"]],
                  str(mp4_dir / f"{t}.{VIEW}.mp4")) for t in ready]
        failed = {}
        with mp.Pool(args.jobs) as pool:
            for n, (t, err) in enumerate(pool.imap_unordered(encode, tasks), 1):
                if err:
                    failed[t] = err
                if n % 100 == 0 or n == len(tasks):
                    print(f"[인코딩] {cam} {n}/{len(tasks)}", flush=True)
        for t, err in list(failed.items())[:5]:
            print(f"[실패] {t}: {err}", flush=True)
        clips = [t for t in ready if t not in failed]
        # 이미 만든 clips.json 이 있으면 클립 구성이 같아야 청크 번호가 카메라마다 맞는다.
        if old is not None and not args.limit_clips and list(old["clips"]) != clips:
            print(f"[중단] {cam}: 기존 clips.json 과 클립 구성이 다르다 - 청크를 쓰지 않는다", flush=True)
            summary[cam] = None
            continue
        where = write_chunks(clips, mp4_dir, args.out, VIEW)
        summary[cam] = (clips, where, len(incomplete), len(failed))

    done = {c: v for c, v in summary.items() if v}
    if old is None and done:
        cam, (clips, where, inc, fail) = next(iter(done.items()))
        doc = {"_meta": {"source": "OpenScene v1.1 test", "camera": cam, "view": view_of(cam),
                         "fps": FPS, "min_frames": args.min_frames,
                         "scenes_total": len(scenes), "clips": len(clips),
                         "incomplete": inc, "failed": fail},
               "clips": {t: {"log": keep[t]["log"], "scene_name": keep[t]["scene_name"],
                             "map_location": keep[t]["map_location"], "chunk": where[t],
                             "n_frames": len(keep[t]["frames"]),
                             "start_us": keep[t]["frames"][0][0],
                             "end_us": keep[t]["frames"][-1][0],
                             "tokens": [f[1] for f in keep[t]["frames"]]} for t in clips}}
    else:
        doc = old
    if doc is not None:
        views = doc["_meta"].setdefault("views", {})
        if "view" in doc["_meta"]:
            views.setdefault(doc["_meta"]["view"], doc["_meta"]["camera"])
        for cam in done:
            views[view_of(cam)] = cam
        tmp = args.out / "clips.json.tmp"
        tmp.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, cj)
        write_ego(keep, list(doc["clips"]), args.out)
    for cam, v in summary.items():
        print(f"[완료] {cam}: " + ("중단" if v is None else
              f"클립 {len(v[0])}개 (프레임 모자람 {v[2]}, 인코딩 실패 {v[3]})"), flush=True)
    print(f"[완료] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
