#!/usr/bin/env python3
"""중단된 클립 추론 실행을 이어받기 위한 결과 CSV 정리.

무엇이 "끝난 클립" 인가:
  CSV 에 행이 있다고 끝난 것이 아니다. 영상을 한 프레임도 못 읽은 클립은
  n_frames=0, observation="(no frames)" 인 행으로 기록되고 verdict 칸에는
  기본값 Normal 이 찍힌다 - 모델에 입력조차 되지 않았는데 결과처럼 보인다.
  실측(20260904_124936_video8b): 163,084행 중 136,059행이 이 상태였다. NAS
  마운트가 끊긴 동안 기록된 것이라 다시 읽으면 대부분 살아난다.

  디코딩이 중간에 끊겨 일부 프레임만 읽힌 클립도 같은 방식으로 기록된다 -
  run_clip_inference 가 요청 장수와 대조해 모자라면 모델에 넣지 않고
  n_frames=0, observation="(partial frames 17/40)" 으로 남긴다. 그래서
  아래 규칙 하나로 두 경우가 모두 재시도 대상이 된다.

  그래서 n_frames 가 양의 정수인 행만 "끝남" 으로 친다. 그 밖의 행 -
  n_frames=0, 칸 수가 헤더와 다른 행(프로세스가 쓰다가 죽어 잘린 마지막
  줄), uuid 가 빈 행 - 은 전부 다시 돌릴 대상이다.

  parse_ok=0 (모델은 돌았지만 JSON 을 못 읽음) 은 끝난 것으로 친다. 그건
  입력은 제대로 들어간 실제 추론 결과이고, 다시 돌려도 결정적 디코딩이라
  같은 답이 나온다.

왜 실패 행을 지워야 하는가:
  이어받기는 실패한 클립을 다시 돌려 새 행을 덧붙인다. 옛 실패 행을 남겨
  두면 같은 uuid 가 두 번 기록되고, 집계가 uuid 로 중복을 걸러낼 때 어느
  행이 살아남는지는 파일 순서에 달린다(merge_shards 는 keep="first" 라
  옛 실패 행이 이긴다).

어떻게 안전하게 지우는가:
  1. 파일을 건드리기 전에 이 폴더에 쓰고 있는 프로세스가 없는지 본다.
  2. 모든 결과 CSV 의 헤더가 서로 같은지(그리고 주어지면 지금 코드의 열
     구성과 같은지) 본다 - 다르면 아무것도 바꾸지 않고 멈춘다.
  3. 바꿀 파일을 resume_backup_<시각>/ 에 먼저 복사한다.
  4. 남길 행만 같은 폴더의 임시 파일에 쓰고 fsync 한 뒤 os.replace 로
     바꿔 끼운다. 쓰는 도중 죽어도 원본이 반쪽이 되지 않는다.
  5. 바꾼 파일을 다시 읽어 실패 행이 0 이고 남은 행 수가 예상과 같은지
     확인한다.

남길 행은 csv 모듈로 읽고 쓴다. pandas 로 왕복하면 빈 칸이 NaN 이 되고
정수가 실수가 되어, 지우지 않은 행까지 내용이 바뀐다.

Usage:
  python resume_run.py --run-dir results/unlabeled/<run>            # 점검만
  python resume_run.py --run-dir results/unlabeled/<run> --apply    # 정리
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

RESULT_GLOB = "clip_results*.csv"
BACKUP_PREFIX = "resume_backup_"

# 모델 출력이나 센서 서술에 긴 문장이 들어가므로 기본 한도(128KB)로는 부족하다.
csv.field_size_limit(sys.maxsize)


class ResumeError(RuntimeError):
    """이어받기를 진행하면 안 되는 상태. 파일은 아무것도 바뀌지 않았다."""


def result_files(run_dir) -> list[Path]:
    """실행 폴더의 결과 CSV 들. 샤드 파일과 병합본을 모두 포함한다.

    백업은 하위 폴더에 두므로 여기 잡히지 않는다.
    """
    return sorted(Path(run_dir).glob(RESULT_GLOB))


def _is_done(row: list[str], header: list[str], i_uuid: int,
             i_frames: int) -> bool:
    """이 행이 '끝난 클립' 인가 - 칸 수가 맞고, uuid 가 있고, n_frames > 0."""
    if len(row) != len(header):
        return False
    if not row[i_uuid].strip():
        return False
    try:
        return int(float(row[i_frames])) > 0
    except (TypeError, ValueError):
        return False


def _read(path: Path) -> tuple[list[str], list[list[str]]]:
    """(헤더, 행들). 빈 파일이면 ([], [])."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        return [], []
    return rows[0], rows[1:]


def scan(run_dir, expected_columns: list[str] | None = None) -> dict:
    """결과 CSV 들을 읽어 이어받기에 필요한 것을 센다. 파일은 바꾸지 않는다.

    반환:
      header    모든 파일이 공유하는 헤더 (파일이 없으면 None)
      done      끝난 클립의 uuid 집합
      files     [{path, n_rows, n_keep, n_drop, n_dup}]
      n_drop    지울 행 수 (실패 + 잘린 행 + 중복)
    헤더가 파일끼리 다르거나 expected_columns 와 다르면 ResumeError.
    """
    files = result_files(run_dir)
    header = None
    done: set[str] = set()
    report = []
    for p in files:
        h, rows = _read(p)
        if not h:
            report.append({"path": p, "n_rows": 0, "n_keep": 0,
                           "n_drop": 0, "n_dup": 0})
            continue
        if header is None:
            header = h
        elif h != header:
            raise ResumeError(
                f"{p.name} 의 열 구성이 다른 결과 파일과 다릅니다 - 서로 다른 "
                f"코드 버전으로 쓴 파일이 섞여 있습니다.\n"
                f"  {files[0].name}: {len(header)}열\n  {p.name}: {len(h)}열")
        if "uuid" not in h or "n_frames" not in h:
            raise ResumeError(f"{p.name} 에 uuid / n_frames 열이 없습니다 - "
                              f"클립 모드 결과가 아닙니다.")
        i_uuid, i_frames = h.index("uuid"), h.index("n_frames")
        n_keep = n_drop = n_dup = 0
        for r in rows:
            if _is_done(r, h, i_uuid, i_frames):
                u = r[i_uuid].strip()
                if u in done:
                    n_dup += 1          # 이미 다른 행이 끝냈다 - 지운다
                else:
                    done.add(u)
                    n_keep += 1
            else:
                n_drop += 1
        report.append({"path": p, "n_rows": len(rows), "n_keep": n_keep,
                       "n_drop": n_drop, "n_dup": n_dup})

    if header is not None and expected_columns is not None \
            and list(header) != list(expected_columns):
        old, new = set(header), set(expected_columns)
        raise ResumeError(
            "기존 결과 CSV 의 열 구성이 지금 코드와 다릅니다 - 이 실행은 다른 "
            "코드 버전으로 만들어져 이어 쓸 수 없습니다. 이어 쓰면 한 파일 "
            "안에서 열의 뜻이 행마다 달라집니다.\n"
            f"  기존에만 있는 열: {sorted(old - new) or '-'}\n"
            f"  지금에만 있는 열: {sorted(new - old) or '-'}"
            + ("\n  (열 이름은 같고 순서만 다릅니다)" if old == new else ""))
    return {"header": header, "done": done, "files": report,
            "n_drop": sum(f["n_drop"] + f["n_dup"] for f in report)}


def writers_alive(run_dir) -> list[int]:
    """이 폴더에 결과를 쓰고 있는 추론 프로세스의 pid.

    /proc/<pid>/cmdline 에 edge_case_mining 과 이 폴더 경로가 함께 들어 있는
    것을 찾는다. 자기 자신(정리 단계로 불린 프로세스)은 뺀다.
    """
    run_dir = Path(run_dir).resolve()
    needles = {str(run_dir), os.path.relpath(run_dir)}
    me = os.getpid()
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == me:
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", "replace")
        except OSError:
            continue
        if "edge_case_mining" in cmd and any(n in cmd for n in needles):
            out.append(int(d.name))
    return out


def prepare(run_dir, expected_columns: list[str] | None = None,
            log=print) -> dict:
    """실패 행을 안전하게 지운다. 이어받기 전에 한 번만 부른다.

    여러 샤드가 동시에 부르면 안 된다 - 같은 병합본을 동시에 다시 쓰게 된다.
    셸 스크립트는 샤드를 띄우기 전에 한 프로세스로 이것을 부른다.
    """
    run_dir = Path(run_dir)
    alive = writers_alive(run_dir)
    if alive:
        raise ResumeError(
            f"이 폴더에 결과를 쓰고 있는 추론 프로세스가 있습니다 (pid "
            f"{', '.join(map(str, alive))}). 끝나거나 멈춘 뒤에 이어받으세요 - "
            f"쓰는 중인 파일을 정리하면 그 사이에 기록된 행이 사라집니다.")

    info = scan(run_dir, expected_columns)
    if info["n_drop"] == 0:
        log(f"[resume] 지울 행 없음 - 끝난 클립 {len(info['done']):,}개")
        return info

    # 1) 백업. 바꿀 파일만 복사한다.
    stamp = time.strftime("%Y%m%d_%H%M%S")
    bdir = run_dir / f"{BACKUP_PREFIX}{stamp}"
    bdir.mkdir(parents=False, exist_ok=False)
    targets = [f for f in info["files"] if f["n_drop"] + f["n_dup"]]
    for f in targets:
        shutil.copy2(f["path"], bdir / f["path"].name)
    log(f"[resume] 백업 {len(targets)}개 파일 -> {bdir}")

    # 2) 남길 행만 임시 파일에 쓰고 바꿔 끼운다. scan() 과 같은 규칙으로
    #    다시 판정한다 - 파일을 오가며 '끝남' 집합을 이어받아야 파일 사이의
    #    중복도 같은 결과로 걸러진다.
    header = info["header"]
    i_uuid, i_frames = header.index("uuid"), header.index("n_frames")
    seen: set[str] = set()
    for f in info["files"]:
        p = f["path"]
        h, rows = _read(p)
        if not h:
            continue
        keep = []
        for r in rows:
            if _is_done(r, h, i_uuid, i_frames):
                u = r[i_uuid].strip()
                if u not in seen:
                    seen.add(u)
                    keep.append(r)
        if len(keep) == len(rows):
            continue
        tmp = p.with_name(f".{p.name}.resume_tmp")
        with open(tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(h)
            w.writerows(keep)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
        log(f"[resume] {p.name}: {len(rows):,}행 -> {len(keep):,}행 "
            f"(실패/잘림 {f['n_drop']:,}, 중복 {f['n_dup']:,} 삭제)")

    # 3) 확인. 다시 읽어 지울 것이 남지 않았고 끝난 클립 수가 그대로인지.
    after = scan(run_dir, expected_columns)
    if after["n_drop"] or after["done"] != info["done"]:
        raise ResumeError(
            f"정리 후 확인에 실패했습니다 (남은 실패 행 {after['n_drop']}, "
            f"끝난 클립 {len(info['done'])} -> {len(after['done'])}). "
            f"원본은 {bdir} 에 있습니다.")
    log(f"[resume] 정리 완료 - 끝난 클립 {len(after['done']):,}개 유지, "
        f"{info['n_drop']:,}행 삭제")
    return after


def _print_scan(info: dict):
    for f in info["files"]:
        print(f"  {f['path'].name:28} {f['n_rows']:9,}행  "
              f"끝남 {f['n_keep']:9,}  실패/잘림 {f['n_drop']:9,}  "
              f"중복 {f['n_dup']:5,}")
    print(f"  끝난 클립 {len(info['done']):,}개 / 지울 행 {info['n_drop']:,}개")


def main():
    ap = argparse.ArgumentParser(
        description="중단된 클립 추론 실행의 결과 CSV 를 점검/정리한다.")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="실패 행을 실제로 지운다 (백업 후). 없으면 점검만 한다.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not result_files(run_dir):
        sys.exit(f"[error] {run_dir} 에 {RESULT_GLOB} 가 없습니다")
    try:
        if args.apply:
            prepare(run_dir)
        else:
            info = scan(run_dir)
            _print_scan(info)
            print("  (점검만 했습니다. 지우려면 --apply)")
    except ResumeError as e:
        sys.exit(f"[error] {e}")


if __name__ == "__main__":
    main()
