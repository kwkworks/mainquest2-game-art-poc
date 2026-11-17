"""결함이 있던 게임 배경 제거본을 AI 결과로 다시 만든다 (게임 폴더는 건드리지 않는다).

대상: data/game_art/qa_labels.json 에서 game_cut_defect=true 이고 ai_defect=false 인 그림.
형식: 게임의 art/cut 과 같게 — 대상 둘레를 잘라 높이 900px, 투명 배경 webp.
결과: outputs/cut_fixed/<이름>.webp, 전후 비교 outputs/cut_fixed/before_after.png

사용법: python src/fix_cuts.py [--method birefnet-general] [--height 900] [--pad 12]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from bg_remove import apply_alpha

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", default="birefnet-general")
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--pad", type=int, default=12)
    a = ap.parse_args()

    lab = json.loads((ROOT / "data/game_art/qa_labels.json").read_text(encoding="utf-8"))
    targets = [n for n, v in lab.items() if not n.startswith("_") and v["game_cut_defect"] and not v["ai_defect"]]
    out = ROOT / "outputs/cut_fixed"; out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for n in targets:
        orig = Image.open(ROOT / f"data/game_art/original/{n}.webp").convert("RGB")
        alpha = np.asarray(Image.open(ROOT / f"outputs/qa/alpha/{a.method}/{n}.png"))
        cut = apply_alpha(orig, alpha, crop=True, pad=a.pad)
        cut = cut.resize((round(cut.width * a.height / cut.height), a.height), Image.LANCZOS)
        cut.save(out / f"{n}.webp", lossless=False, quality=92)
        pairs.append((n, Image.open(ROOT / f"data/game_art/cut/{n}.webp").convert("RGBA"), cut))

    # 게임처럼 어두운 바탕 위에 올려 전후 비교 (남은 배경 조각이 잘 보이도록)
    T = 300
    f = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", 18) if Path("C:/Windows/Fonts/malgun.ttf").exists() else None
    S = Image.new("RGB", (T * 2, (T + 28) * len(pairs) + 30), (20, 18, 22)); d = ImageDraw.Draw(S)
    d.text((8, 4), "지금 게임에 들어간 것", fill="white", font=f); d.text((T + 8, 4), "AI로 다시 만든 것", fill="white", font=f)
    for r, (n, before, after) in enumerate(pairs):
        y = 30 + r * (T + 28)
        d.text((8, y), n, fill=(200, 200, 200), font=f)
        for c, im in enumerate([before, after]):
            bg = Image.new("RGBA", im.size, (38, 34, 40, 255)); bg.alpha_composite(im)
            im2 = bg.convert("RGB"); im2.thumbnail((T - 8, T - 8))
            S.paste(im2, (c * T + (T - im2.width) // 2, y + 24))
    S.save(out / "before_after.png")
    print(len(pairs), "장 →", out)


if __name__ == "__main__":
    main()
