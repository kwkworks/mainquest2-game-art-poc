"""공개 저장소용 미리보기 그림을 만든다 (outputs/public/).

출시 전 게임 에셋이라 원본은 공개하지 않는다. 이 스크립트는
- 그림을 작게 줄이고 (긴 변 PREVIEW_MAX 이하)
- 그림 이름을 data/public_name_map.json의 공개용 이름으로 바꾸고
- "비공개 게임 에셋 · 미리보기" 표시를 넣는다.

사용법: python tools/public_assets.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from qa_cuts import ai_cut  # noqa: E402

OUT = ROOT / "outputs/public"
MAP = json.loads((ROOT / "data/public_name_map.json").read_text(encoding="utf-8"))
SPEC = json.loads((ROOT / "data/icons.json").read_text(encoding="utf-8"))
MARK = "비공개 게임 에셋 · 미리보기 · 무단 사용 금지"


def font(n: int):
    p = "C:/Windows/Fonts/malgun.ttf"
    return ImageFont.truetype(p, n) if Path(p).exists() else ImageFont.load_default()


def mark(im: Image.Image) -> Image.Image:
    im = im.convert("RGB"); d = ImageDraw.Draw(im); f = font(14)
    w = d.textlength(MARK, font=f)
    d.rectangle((im.width - w - 12, im.height - 22, im.width, im.height), fill=(0, 0, 0))
    d.text((im.width - w - 6, im.height - 20), MARK, fill=(200, 200, 200), font=f)
    return im


def on(rgba: Image.Image, color) -> Image.Image:
    bg = Image.new("RGBA", rgba.size, color); bg.alpha_composite(rgba.convert("RGBA")); return bg.convert("RGB")


def triple(n: str, T: int):
    al = json.loads((ROOT / "data/game_art/align.json").read_text(encoding="utf-8"))[n]
    cut = Image.open(ROOT / f"data/game_art/cut/{n}.webp").convert("RGBA")
    oc = Image.open(ROOT / f"data/game_art/original/{n}.webp").convert("RGB").crop(
        (al["x"], al["y"], al["x"] + al["w"], al["y"] + al["h"])).resize(cut.size)
    ai = oc.convert("RGBA"); ai.putalpha(Image.fromarray((ai_cut(n, "birefnet-general") * 255).astype(np.uint8)))
    ims = [oc, on(cut, (255, 0, 255, 255)), on(ai, (255, 0, 255, 255))]
    for i in ims:
        i.thumbnail((T, T))
    return ims


def defect_examples() -> None:
    # 결함 판정 그림 중 무기 2장, 무기 아닌 것 2장 (IoU가 낮은 순)
    import pandas as pd
    q = pd.read_csv(ROOT / "outputs/qa/qa_scores.csv")
    d = q[q.defect].sort_values("agreement_iou").image.tolist()
    names = [n for n in d if n.startswith("wpn")][:2] + [n for n in d if not n.startswith("wpn")][1:3]
    T, f = 200, font(16)
    S = Image.new("RGB", (T * 3, 28 + (T + 24) * len(names)), "white"); d = ImageDraw.Draw(S)
    for c, h in enumerate(["원본", "게임 배경 제거본", "AI (BiRefNet)"]):
        d.text((c * T + 6, 4), h, fill="black", font=f)
    for r, n in enumerate(names):
        y = 28 + r * (T + 24)
        d.text((6, y), MAP[n], fill="black", font=f)
        for c, im in enumerate(triple(n, T)):
            S.paste(im, (c * T + (T - im.width) // 2, y + 22))
    mark(S).save(OUT / "bg_defect_examples.png")


def before_after() -> None:
    lab = json.loads((ROOT / "data/game_art/qa_labels.json").read_text(encoding="utf-8"))
    names = [n for n, v in lab.items() if not n.startswith("_") and v["game_cut_defect"] and not v["ai_defect"]]
    T, cols, f = 170, 3, font(15)
    rows = (len(names) + cols - 1) // cols
    S = Image.new("RGB", (cols * T * 2, 28 + rows * (T + 22)), (20, 18, 22)); d = ImageDraw.Draw(S)
    for c in range(cols):
        d.text((c * 2 * T + 6, 4), "지금 게임", fill="white", font=f); d.text((c * 2 * T + T + 6, 4), "AI 교체본", fill="white", font=f)
    for k, n in enumerate(names):
        r, c = divmod(k, cols); x, y = c * 2 * T, 28 + r * (T + 22)
        d.text((x + 6, y), MAP[n], fill=(190, 190, 190), font=f)
        for j, p in enumerate([ROOT / f"data/game_art/cut/{n}.webp", ROOT / f"outputs/cut_fixed/{n}.webp"]):
            im = on(Image.open(p), (40, 36, 42, 255)); im.thumbnail((T - 6, T - 6))
            S.paste(im, (x + j * T + (T - im.width) // 2, y + 20))
    mark(S).save(OUT / "bg_before_after.png")


def failure_zoom() -> None:
    im = Image.open(ROOT / "outputs/bg_eval/failure_zoom.png"); im.thumbnail((900, 900))
    mark(im).save(OUT / "bg_failure_zoom.png")


def icons_final(cand: str) -> None:
    import pandas as pd
    log = pd.read_csv(ROOT / f"outputs/final_log_{cand}.csv").set_index("icon")
    T, f = 150, font(14)
    slot = Image.open(ROOT / "data/game_ui_slot.webp").convert("RGBA").resize((T, T))
    S = Image.new("RGB", (T * 5, (T + 26) * 2), (18, 16, 20)); d = ImageDraw.Draw(S)
    for k, ic in enumerate(SPEC["icons"]):
        x, y = (k % 5) * T, (k // 5) * (T + 26)
        S.paste(slot, (x, y), slot)
        ico = Image.open(ROOT / f"outputs/icons_game/{cand}/{ic['id']}.png").convert("RGBA").resize((round(T * 0.72),) * 2, Image.LANCZOS)
        S.paste(ico, (x + (T - ico.width) // 2, y + (T - ico.height) // 2), ico)
        rv = not bool(log.loc[ic["id"], "auto_adopted"])
        d.text((x + 6, y + T + 4), ic["name"] + (" · 검토" if rv else ""), fill=(255, 170, 90) if rv else "white", font=f)
    mark(S).save(OUT / f"icons_final_{cand}.png")


def icons_candidates(cand: str) -> None:
    T, lw, f = 110, 110, font(13)
    ids = [i["id"] for i in SPEC["icons"]]
    S = Image.new("RGB", (lw + T * 4, 20 + T * len(ids)), (24, 22, 26)); d = ImageDraw.Draw(S)
    for s in range(4):
        d.text((lw + s * T + 4, 2), f"seed {s}", fill="white", font=f)
    for r, ic in enumerate(SPEC["icons"]):
        d.text((4, 20 + r * T + 4), ic["name"], fill="white", font=f)
        for s in range(4):
            im = Image.open(ROOT / f"outputs/icons_raw/{cand}/{ic['id']}_s{s}.png").convert("RGB").resize((T - 2, T - 2))
            S.paste(im, (lw + s * T + 1, 20 + r * T + 1))
    mark(S).save(OUT / f"icons_candidates_{cand}.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    defect_examples(); before_after(); failure_zoom()
    for c in ["sdxl", "sdxl_ipa"]:
        icons_final(c); icons_candidates(c)
    for p in sorted(OUT.glob("*.png")):
        print(p.name, Image.open(p).size)


if __name__ == "__main__":
    main()
