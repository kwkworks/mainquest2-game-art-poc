"""아이콘 생성 (로컬 GPU, 외부 전송 없음).

후보
- sdxl      : SDXL base, 프롬프트만으로 화풍을 요청 (범용)
- sdxl_ipa  : SDXL base + IP-Adapter. 게임 기준 그림(master_style_A)을 '화풍 참조'로 넣는다.
              InstantStyle 방식으로 화풍 블록에만 참조를 걸어, 그림 속 인물(내용)은 따라오지 않게 한다.

시드를 고정하므로 같은 설정이면 같은 그림이 나온다. 이미 있는 파일은 덮어쓰지 않는다.

사용법: python src/gen_icons.py --candidate sdxl_ipa [--seeds 0 1 2 3] [--ids face_skull ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE = "stabilityai/stable-diffusion-xl-base-1.0"


def load_pipe(candidate: str, ip_scale: float):
    from diffusers import StableDiffusionXLPipeline

    pipe = StableDiffusionXLPipeline.from_pretrained(BASE, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    if candidate == "sdxl_ipa":
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter_sdxl_vit-h.safetensors", image_encoder_folder="models/image_encoder")
        # InstantStyle: 화풍을 담당하는 up_blocks.0.attentions.1 에만 참조를 건다
        pipe.set_ip_adapter_scale({"up": {"block_0": [0.0, ip_scale, 0.0]}})
    pipe.enable_model_cpu_offload()
    return pipe


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", choices=["sdxl", "sdxl_ipa"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--ids", nargs="*", default=None, help="일부 아이콘만 (기본: 전부)")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--ip-scale", type=float, default=0.8, help="sdxl_ipa에서 화풍 참조 세기")
    ap.add_argument("--style-ref", default="data/style_ref/master_style_A.webp")
    ap.add_argument("--out", default="outputs/icons_raw")
    ap.add_argument("--prompts", default="data/prompts.private.json",
                    help="생성 프롬프트 파일 (비공개, git에 올리지 않음). 형식은 data/prompts.example.json")
    a = ap.parse_args()

    spec = json.loads((ROOT / "data/icons.json").read_text(encoding="utf-8"))
    pp = ROOT / a.prompts
    if not pp.exists():
        raise SystemExit(f"{a.prompts} 가 없습니다. data/prompts.example.json 을 복사해 프롬프트를 채우세요.")
    P = json.loads(pp.read_text(encoding="utf-8"))
    spec.update({k: P[k] for k in ("style", "layout", "negative")})
    for ic in spec["icons"]:
        ic["prompt"] = P["prompts"][ic["id"]]
    icons = [i for i in spec["icons"] if not a.ids or i["id"] in a.ids]
    out_dir = ROOT / a.out / a.candidate
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "outputs/gen_log.csv"
    new_log = not log_path.exists()
    ref = Image.open(ROOT / a.style_ref).convert("RGB") if a.candidate == "sdxl_ipa" else None

    pipe = None
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_log:
            w.writerow(["candidate", "icon", "seed", "seconds", "steps", "guidance", "ip_scale", "gpu", "file"])
        for ic in icons:
            for seed in a.seeds:
                dst = out_dir / f"{ic['id']}_s{seed}.png"
                if dst.exists():
                    continue
                if pipe is None:
                    t0 = time.perf_counter(); pipe = load_pipe(a.candidate, a.ip_scale)
                    print(f"모델 불러오기 {time.perf_counter() - t0:.0f}초")
                kw = dict(prompt=f"{spec['style']} {ic['prompt']}. {spec['layout']}", negative_prompt=spec["negative"],
                          width=a.size, height=a.size, num_inference_steps=a.steps, guidance_scale=a.guidance,
                          generator=torch.Generator("cpu").manual_seed(seed))
                if ref is not None:
                    kw["ip_adapter_image"] = ref
                torch.cuda.synchronize(); t0 = time.perf_counter()
                img = pipe(**kw).images[0]
                torch.cuda.synchronize(); sec = time.perf_counter() - t0
                img.save(dst)
                w.writerow([a.candidate, ic["id"], seed, f"{sec:.1f}", a.steps, a.guidance,
                            a.ip_scale if ref is not None else "", torch.cuda.get_device_name(0), dst.relative_to(ROOT).as_posix()])
                f.flush()
                print(f"{a.candidate} {ic['id']} s{seed}: {sec:.1f}초")


if __name__ == "__main__":
    main()
