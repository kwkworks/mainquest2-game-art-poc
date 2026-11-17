"""생성한 아이콘을 SigLIP 2로 잰다 (로컬 실행).

1) 알아볼 수 있는가: 아이콘 그림을 한국어 이름 10개(label) 중 하나로 고르게 한다 (10지선다).
   그림이 무엇인지 모델이 맞히면 '알아볼 수 있음'의 근거 하나로 쓴다. 사람 판정과 함께 본다.
2) 화풍이 맞는가: 게임에 이미 들어간 ChatGPT 아이템 그림 9장(무기 8 + 주사위 1, data/style_ref)의
   SigLIP 임베딩 평균과의 코사인 유사도. 기준선 범위는 9장 각각을 나머지 8장 평균과 견준 값이다.

점수는 반올림하지 않고 CSV에 그대로 남긴다.
사용법: python src/eval_icons.py [--dirs outputs/icons_raw/sdxl outputs/icons_raw/sdxl_ipa]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
MODEL = "google/siglip2-base-patch16-224"


class Siglip:
    def __init__(self, device: str):
        from transformers import AutoModel, AutoProcessor
        self.m = AutoModel.from_pretrained(MODEL).to(device).eval()
        self.p = AutoProcessor.from_pretrained(MODEL)
        self.d = device

    @staticmethod
    def _vec(out) -> torch.Tensor:
        # transformers 버전에 따라 텐서 또는 pooler_output을 가진 객체를 돌려준다
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def img(self, ims: list[Image.Image]) -> torch.Tensor:
        x = self.p(images=[i.convert("RGB") for i in ims], return_tensors="pt").to(self.d)
        e = self._vec(self.m.get_image_features(**x))
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.no_grad()
    def txt(self, texts: list[str]) -> torch.Tensor:
        x = self.p(text=texts, padding="max_length", max_length=64, return_tensors="pt").to(self.d)
        e = self._vec(self.m.get_text_features(**x))
        return torch.nn.functional.normalize(e, dim=-1)

    def logits(self, ie: torch.Tensor, te: torch.Tensor) -> torch.Tensor:
        return ie @ te.T * self.m.logit_scale.exp() + self.m.logit_bias


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", default=["outputs/icons_raw/sdxl", "outputs/icons_raw/sdxl_ipa"])
    ap.add_argument("--template", default="{} 그림", help="한국어 질의 틀")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    S = Siglip(dev)
    spec = json.loads((ROOT / "data/icons.json").read_text(encoding="utf-8"))
    ids = [i["id"] for i in spec["icons"]]
    te = S.txt([a.template.format(i["label"]) for i in spec["icons"]])

    refs = sorted((ROOT / "data/style_ref").glob("*.webp"))
    refs = [p for p in refs if p.stem != "master_style_A"]
    re_ = S.img([Image.open(p) for p in refs])
    centroid = torch.nn.functional.normalize(re_.mean(0), dim=0)

    out = ROOT / "outputs"
    with open(out / "style_baseline.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ref_image", "cos_to_others_mean"])
        for k, p in enumerate(refs):
            others = torch.nn.functional.normalize(torch.cat([re_[:k], re_[k + 1:]]).mean(0), dim=0)
            w.writerow([p.name, float(re_[k] @ others)])

    rows = []
    for d in a.dirs:
        files = sorted((ROOT / d).glob("*.png"))
        for p in files:
            icon_id, seed = p.stem.rsplit("_s", 1)
            ie = S.img([Image.open(p)])
            prob = torch.softmax(S.logits(ie, te)[0], dim=0)   # 10개 이름 중 고르기
            sig = torch.sigmoid(S.logits(ie, te)[0])          # 이름 하나하나와 따로 잰 점수
            k = ids.index(icon_id)
            rows.append({"candidate": Path(d).name, "icon": icon_id, "seed": int(seed),
                         "pred": ids[int(prob.argmax())], "correct": int(prob.argmax()) == k,
                         "p_correct": float(prob[k]), "sigmoid_correct": float(sig[k]),
                         "style_cos": float(ie[0] @ centroid)})
    with open(out / "icon_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    base = pd.read_csv(out / "style_baseline.csv")
    print("기준선(ChatGPT 아이템 9장) 화풍 유사도: 최저", base.cos_to_others_mean.min(), "평균", base.cos_to_others_mean.mean())
    g = df.groupby("candidate")
    print(pd.DataFrame({
        "정답률(장)": g.correct.mean(),
        "알아본 아이콘 수(시드 4장 중 1장 이상)": g.apply(lambda x: int(x.groupby("icon").correct.any().sum())),
        "화풍 유사도 평균": g.style_cos.mean(),
        "화풍 유사도 최저": g.style_cos.min(),
        "기준선 최저 이상 비율": g.style_cos.apply(lambda s: float((s >= base.cos_to_others_mean.min()).mean())),
    }).to_string())


if __name__ == "__main__":
    main()
