# 복원 지점: "후보만 골라 다시 묻기(recheck)" 적용 직전 (2026-10-13)

- HEAD.txt         : 이 시점의 git 커밋 (556ed31)
- working_tree.diff: 그 커밋 위에 있던 커밋 안 된 변경
- code.tar.gz      : 이 시점의 *.py, *.sh, gui/index.html, scene_category_3.0.json, config.py

복원: 프로젝트 폴더에서
    tar xzf .snapshots/20261013_before_recheck/code.tar.gz
(recheck 로 새로 생긴 파일 recheck_missed.py 는 따로 지운다)
