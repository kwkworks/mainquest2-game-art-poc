"""배경 제거 품질 검사기 (검사·판독 유형).

게임에 이미 들어간 배경 제거본(data/game_art/cut)을 AI 배경 제거 결과와 견주어,
서로 많이 다른 그림을 '검토 필요'로 표시한다. 최종 판정은 사람이 한다.

단계
  python src/qa_cuts.py alphas   # AI 알파를 만들어 저장 (outputs/qa/alpha/, 다시 돌려도 덮어쓰지 않음)
  python src/qa_cuts.py sheets   # 사람이 판정할 비교 그림 (점수 없이, 무작위 순서)
  python src/qa_cuts.py report   # data/game_art/qa_labels.json(사람 판정)과 견주어 임계값별 정밀도·재현율
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from bg_remove import alpha_for
from eval_bg import crop_alpha, gt_alpha, iou

ROOT = Path(__file__).resolve().parents[1]
ORIG, CUT = ROOT / "data/game_art/original", ROOT / "data/game_art/cut"
QA = ROOT / "outputs/qa"


def names() -> list[str]:
    return sorted(p.stem for p in CUT.glob("*.webp"))


def align() -> dict:
    return json.loads((ROOT / "data/game_art/align.json").read_text(encoding="utf-8"))


def ai_cut(n: str, method: str) -> np.ndarray:
    """AI 알파를 게임 배경 제거본과 같은 영역·크기로 맞춘 것 (0~1)."""
    a = np.asarray(Image.open(QA / "alpha" / method / f"{n}.png"))
    return crop_alpha(a, align()[n], Image.open(CUT / f"{n}.webp").size)


def on_magenta(rgba: Image.Image) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, (255, 0, 255, 255)); bg.alpha_composite(rgba)
    return bg.convert("RGB")


def cmd_alphas(method: str) -> None:
    d = QA / "alpha" / method; d.mkdir(parents=True, exist_ok=True)
    for n in names():
        if (d / f"{n}.png").exists():
            continue
        Image.fromarray(alpha_for(Image.open(ORIG / f"{n}.webp").convert("RGB"), method)).save(d / f"{n}.png")
        print("알파", n)


def cmd_sheets(method: str, seed: int) -> None:
    """원본 | 게임 배경 제거본 | AI 배경 제거본. 점수를 보여 주지 않고 순서도 섞어서, 판정이 점수에 끌려가지 않게 한다."""
    al = align(); order = names(); random.Random(seed).shuffle(order)
    T, per = 260, 8
    f = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 16) if Path("C:/Windows/Fonts/malgun.ttf").exists() else None
    out = QA / "label_sheets"; out.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(order), per):
        chunk = order[k:k + per]
        S = Image.new("RGB", (T * 3, (T + 22) * len(chunk)), "white"); d = ImageDraw.Draw(S)
        for r, n in enumerate(chunk):
            b = al[n]; cut = Image.open(CUT / f"{n}.webp").convert("RGBA")
            oc = Image.open(ORIG / f"{n}.webp").convert("RGB").crop((b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])).resize(cut.size)
            ai = oc.convert("RGBA"); ai.putalpha(Image.fromarray((ai_cut(n, method) * 255).astype(np.uint8)))
            d.text((4, r * (T + 22) + 2), f"{k + r + 1}. {n}", fill="black", font=f)
            for c, im in enumerate([oc, on_magenta(cut), on_magenta(ai)]):
                im = im.copy(); im.thumbnail((T, T)); S.paste(im, (c * T, r * (T + 22) + 22))
        S.save(out / f"sheet_{k // per + 1}.png")
    print("저장:", out, "(열: 원본 | 게임 배경 제거본 | AI)")


def cmd_report(method: str) -> None:
    lab = json.loads((ROOT / "data/game_art/qa_labels.json").read_text(encoding="utf-8"))
    rows = []
    for n in names():
        gt = gt_alpha(Image.open(CUT / f"{n}.webp"))
        rows.append({"image": n, "agreement_iou": iou(ai_cut(n, method), gt), "defect": bool(lab[n]["game_cut_defect"])})
    df = pd.DataFrame(rows).sort_values("agreement_iou")
    df.to_csv(QA / "qa_scores.csv", index=False, encoding="utf-8")
    pos = int(df.defect.sum())
    table = []
    for t in np.round(np.arange(0.80, 0.981, 0.01), 2):
        flag = df.agreement_iou < t
        tp, fp = int((flag & df.defect).sum()), int((flag & ~df.defect).sum())
        table.append({"threshold": t, "flagged": int(flag.sum()), "caught_defects": tp, "false_flags": fp,
                      "recall": tp / pos if pos else float("nan"), "precision": tp / (tp + fp) if tp + fp else float("nan")})
    tb = pd.DataFrame(table); tb.to_csv(QA / "qa_threshold_table.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False)); print(); print(tb.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["alphas", "sheets", "report"])
    ap.add_argument("--method", default="birefnet-general")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    {"alphas": lambda: cmd_alphas(a.method), "sheets": lambda: cmd_sheets(a.method, a.seed),
     "report": lambda: cmd_report(a.method)}[a.cmd]()


if __name__ == "__main__":
    main()
