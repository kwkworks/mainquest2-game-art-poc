"""PoC 끝단: 생성한 후보에서 아이콘마다 한 장을 고르고, 배경을 지워 게임용 아이콘으로 저장한다.

고르는 방법 (--pick)
- siglip : icon_scores.csv에서 정답 이름 점수(p_correct)가 가장 높은 시드 (자동)
           단, 4장 모두 10지선다를 틀렸거나 가장 높은 p_correct가 --min-p 아래면 자동으로 채택하지 않고
           '검토 필요'로 넘긴다 (레슨: 확신이 낮은 건은 사람에게). 이때도 비교용으로 가장 높은 시드를 만들어 둔다.
- manual : data/picks.json에 사람이 적은 시드 ({"face_skull": 2, ...})

결과
- outputs/icons_game/<후보>/<id>.png  : 256x256 RGBA (투명 배경, 가운데 정렬)
- outputs/sheets/candidates_<후보>.png : 아이콘 10종 × 시드 4장 비교표
- outputs/sheets/final_on_game_bg_<후보>.png  : 게임의 어두운 칸 위에 올려 본 최종 아이콘

사용법: python src/make_icons.py --candidate sdxl_ipa --bg birefnet-general [--pick siglip]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from bg_remove import METHODS, alpha_for, apply_alpha

ROOT = Path(__file__).resolve().parents[1]


def font(size: int):
    for p in ["C:/Windows/Fonts/malgun.ttf", "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"]:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def fit(im: Image.Image, size: int, margin: float = 0.08) -> Image.Image:
    """투명 아이콘을 정사각형 가운데에 맞춘다."""
    im = im.copy()
    inner = round(size * (1 - 2 * margin))
    im.thumbnail((inner, inner), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.alpha_composite(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def candidates_sheet(spec, cand: str, seeds: list[int], scores: pd.DataFrame | None) -> Image.Image:
    T, pad, lw = 200, 24, 150
    f = font(16)
    sheet = Image.new("RGB", (lw + T * len(seeds), pad + T * len(spec["icons"])), (24, 22, 26))
    d = ImageDraw.Draw(sheet)
    for c, s in enumerate(seeds):
        d.text((lw + c * T + 6, 4), f"seed {s}", fill="white", font=f)
    for r, ic in enumerate(spec["icons"]):
        y = pad + r * T
        d.text((6, y + 6), f"{ic['name']}\n{ic['group']}", fill="white", font=f)
        for c, s in enumerate(seeds):
            p = ROOT / "outputs/icons_raw" / cand / f"{ic['id']}_s{s}.png"
            if not p.exists():
                continue
            sheet.paste(Image.open(p).convert("RGB").resize((T - 4, T - 4)), (lw + c * T + 2, y + 2))
            if scores is not None:
                row = scores[(scores.candidate == cand) & (scores.icon == ic["id"]) & (scores.seed == s)]
                if len(row):
                    ok = bool(row.correct.iloc[0])
                    d.text((lw + c * T + 6, y + T - 24), f"{'O' if ok else 'X'} p={row.p_correct.iloc[0]:.2f}",
                           fill=(120, 255, 120) if ok else (255, 110, 110), font=f)
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="sdxl_ipa")
    ap.add_argument("--bg", default="birefnet-general", choices=METHODS)
    ap.add_argument("--pick", default="siglip", choices=["siglip", "manual"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--min-p", type=float, default=0.5, help="자동 채택에 필요한 정답 이름 점수(10지선다 확률)의 최저값")
    a = ap.parse_args()

    spec = json.loads((ROOT / "data/icons.json").read_text(encoding="utf-8"))
    sp = ROOT / "outputs/icon_scores.csv"
    scores = pd.read_csv(sp) if sp.exists() else None
    sheets = ROOT / "outputs/sheets"; sheets.mkdir(parents=True, exist_ok=True)
    for cand in sorted({p.name for p in (ROOT / "outputs/icons_raw").iterdir() if p.is_dir()}):
        candidates_sheet(spec, cand, a.seeds, scores).save(sheets / f"candidates_{cand}.png")

    review: dict = {}
    if a.pick == "manual":
        picks = json.loads((ROOT / "data/picks.json").read_text(encoding="utf-8"))
    else:
        s = scores[scores.candidate == a.candidate]
        picks, review = {}, {}
        for i, g in s.groupby("icon"):
            top = g.sort_values("p_correct", ascending=False).iloc[0]
            picks[i] = int(top.seed)
            if not g.correct.any() or top.p_correct < a.min_p:
                review[i] = f"최고 p_correct={top.p_correct:.3f}, 10지선다 정답 {int(g.correct.sum())}/{len(g)}장"

    out = ROOT / "outputs/icons_game" / a.candidate; out.mkdir(parents=True, exist_ok=True)
    finals, log = [], []
    for ic in spec["icons"]:
        seed = picks[ic["id"]]
        raw = Image.open(ROOT / "outputs/icons_raw" / a.candidate / f"{ic['id']}_s{seed}.png").convert("RGB")
        t0 = time.perf_counter()
        alpha = alpha_for(raw, a.bg)
        icon = fit(apply_alpha(raw, alpha), a.size)
        sec = time.perf_counter() - t0
        icon.save(out / f"{ic['id']}.png")
        finals.append((ic, icon))
        log.append({"icon": ic["id"], "seed": seed, "pick": a.pick, "auto_adopted": ic["id"] not in review,
                    "review_reason": review.get(ic["id"], ""), "bg": a.bg, "bg_seconds": sec})
    pd.DataFrame(log).to_csv(ROOT / f"outputs/final_log_{a.candidate}.csv", index=False, encoding="utf-8")

    # 게임 화면처럼 어두운 칸 위에 올려 보기 (배경이 덜 지워졌으면 네모 테두리가 보인다)
    T, f = 220, font(18)
    slot = Image.open(ROOT / "data/game_ui_slot.webp").convert("RGBA").resize((T, T)) if (ROOT / "data/game_ui_slot.webp").exists() else None
    sheet = Image.new("RGB", (T * 5, (T + 34) * 2), (18, 16, 20))
    d = ImageDraw.Draw(sheet)
    for k, (ic, icon) in enumerate(finals):
        x, y = (k % 5) * T, (k // 5) * (T + 34)
        if slot is not None:
            sheet.paste(slot, (x, y), slot)
        ico = icon.resize((round(T * 0.72),) * 2, Image.LANCZOS)
        sheet.paste(ico, (x + (T - ico.width) // 2, y + (T - ico.height) // 2), ico)
        tag = " · 검토 필요" if ic["id"] in review else ""
        d.text((x + 8, y + T + 6), f"{ic['name']} ({ic['group'].split(' · ')[-1]}){tag}",
               fill=(255, 170, 90) if tag else "white", font=f)
    sheet.save(sheets / f"final_on_game_bg_{a.candidate}.png")
    print("저장:", out, sheets)


if __name__ == "__main__":
    main()
