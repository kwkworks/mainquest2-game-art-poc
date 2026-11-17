"""배경 제거 방법 비교 (실제 게임 그림 31장).

정답: 게임에 실제로 넣어 쓰고 있는 배경 제거본 (data/game_art/cut).
  - 원본에서 대상 둘레를 잘라 높이 900px로 키우거나 줄인 그림이다.
  - 그래서 먼저 원본의 어느 영역을 잘랐는지 찾는다 (align.json에 저장해 두고 다시 쓴다).
채점: 각 방법이 원본에 만든 알파를 같은 영역으로 잘라 정답 알파와 비교한다.
  - IoU: 알파 0.5 기준으로 대상/배경을 나눈 뒤 겹친 비율 (1.0이 완전 일치)
  - MAE: 알파 차이의 평균 (0~1, 0이 완전 일치). 반투명 경계까지 본다.

사용법: python src/eval_bg.py [--methods colorkey u2net ...] [--thresh 28] [--feather 12]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from bg_remove import METHODS, alpha_for

ROOT = Path(__file__).resolve().parents[1]
ORIG = ROOT / "data/game_art/original"
CUT = ROOT / "data/game_art/cut"
OUT = ROOT / "outputs/bg_eval"


def find_box(orig: Image.Image, cut: Image.Image) -> dict:
    """정답 그림이 원본의 어느 사각형에서 왔는지 찾는다 (여러 배율로 템플릿 매칭)."""
    o = cv2.cvtColor(np.asarray(orig.convert("RGB")), cv2.COLOR_RGB2GRAY).astype(np.float32)
    c_rgba = np.asarray(cut.convert("RGBA"))
    c = cv2.cvtColor(c_rgba[..., :3], cv2.COLOR_RGB2GRAY).astype(np.float32)
    m = (c_rgba[..., 3] == 255).astype(np.float32)
    ch, cw = c.shape
    best = (-1.0, None)
    # 1단계: 1/4 크기에서 거칠게
    q = 4
    o_s = cv2.resize(o, (o.shape[1] // q, o.shape[0] // q), interpolation=cv2.INTER_AREA)
    for bh in range(int(ch * 0.4), min(o.shape[0], int(ch * 2.0)) + 1, 4):
        s = bh / ch
        tw, th = round(cw * s / q), round(ch * s / q)
        if tw < 8 or th < 8 or tw > o_s.shape[1] or th > o_s.shape[0]:
            continue
        t = cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)
        tm = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
        r = cv2.matchTemplate(o_s, t, cv2.TM_CCORR_NORMED, mask=tm)
        r = np.nan_to_num(r, nan=-1, posinf=-1, neginf=-1)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, (s, loc[0] * q, loc[1] * q))
    # 2단계: 원래 크기에서 근처만 촘촘하게
    s0, x0, y0 = best[1]
    best = (-1.0, None)
    for bh in range(round(ch * s0) - 6, round(ch * s0) + 7):
        s = bh / ch
        tw, th = round(cw * s), round(ch * s)
        xa, ya = max(0, x0 - 12), max(0, y0 - 12)
        xb, yb = min(o.shape[1], x0 + tw + 12), min(o.shape[0], y0 + th + 12)
        region = o[ya:yb, xa:xb]
        if region.shape[0] < th or region.shape[1] < tw:
            continue
        t = cv2.resize(c, (tw, th), interpolation=cv2.INTER_AREA)
        tm = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
        r = np.nan_to_num(cv2.matchTemplate(region, t, cv2.TM_CCORR_NORMED, mask=tm), nan=-1, posinf=-1, neginf=-1)
        _, v, _, loc = cv2.minMaxLoc(r)
        if v > best[0]:
            best = (v, (xa + loc[0], ya + loc[1], tw, th))
    v, (x, y, w, h) = best
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "score": float(v)}


def gt_alpha(cut: Image.Image) -> np.ndarray:
    return np.asarray(cut.convert("RGBA"))[..., 3].astype(np.float32) / 255


def crop_alpha(alpha: np.ndarray, box: dict, size: tuple[int, int]) -> np.ndarray:
    x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    a = Image.fromarray(alpha).crop((x, y, x + w, y + h)).resize(size, Image.BILINEAR)
    return np.asarray(a, dtype=np.float32) / 255


def iou(a: np.ndarray, b: np.ndarray) -> float:
    A, B = a >= 0.5, b >= 0.5
    u = (A | B).sum()
    return float((A & B).sum() / u) if u else 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--thresh", type=float, default=28.0, help="colorkey 배경 거리 기준")
    ap.add_argument("--feather", type=float, default=12.0, help="colorkey 경계 부드럽게")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    align_path = ROOT / "data/game_art/align.json"
    align = json.loads(align_path.read_text(encoding="utf-8")) if align_path.exists() else {}
    names = sorted(p.stem for p in CUT.glob("*.webp"))
    for n in names:
        if n not in align:
            align[n] = find_box(Image.open(ORIG / f"{n}.webp"), Image.open(CUT / f"{n}.webp"))
            print(f"정렬 {n}: {align[n]}")
    align_path.write_text(json.dumps(align, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = []
    for method in a.methods:
        kw = {"thresh": a.thresh, "feather": a.feather} if method == "colorkey" else {}
        alpha_for(Image.new("RGB", (64, 64), "gray"), method, **kw)  # 모델 불러오기 시간은 빼고 잰다
        for n in names:
            orig, cut = Image.open(ORIG / f"{n}.webp"), Image.open(CUT / f"{n}.webp")
            t0 = time.perf_counter()
            alpha = alpha_for(orig, method, **kw)
            sec = time.perf_counter() - t0
            pred = crop_alpha(alpha, align[n], cut.size)
            gt = gt_alpha(cut)
            rows.append({"method": method, "image": n, "iou": iou(pred, gt), "mae": float(np.abs(pred - gt).mean()),
                         "seconds": sec, "align_score": align[n]["score"]})
            # 눈으로 확인할 그림: 정답과 다른 곳을 색으로 (빨강: 지웠어야 하는데 남김, 파랑: 남겼어야 하는데 지움)
            d = Path(OUT / method); d.mkdir(exist_ok=True)
            base = np.asarray(cut.convert("RGB"), dtype=np.float32) * 0.35
            P, G = pred >= 0.5, gt >= 0.5
            base[P & ~G] = [255, 40, 40]; base[~P & G] = [40, 120, 255]
            Image.fromarray(base.astype(np.uint8)).save(d / f"{n}_diff.png")
        print(method, "끝")

    with open(OUT / "bg_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = df.groupby("method").agg(iou_mean=("iou", "mean"), iou_min=("iou", "min"), mae_mean=("mae", "mean"),
                                    sec_mean=("seconds", "mean"), n_iou_ge_095=("iou", lambda s: int((s >= 0.95).sum())))
    summ["n"] = len(names)
    summ = summ.loc[[m for m in a.methods]]
    summ.to_csv(OUT / "bg_summary.csv", encoding="utf-8")
    print(summ.to_string())


if __name__ == "__main__":
    main()
