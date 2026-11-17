"""배경 제거 방법 모음.

- colorkey: 기준선. 그림 테두리의 배경색을 재고, 그 색과 가까우면서 테두리와 이어진 영역을 배경으로 본다 (AI 없음).
- u2net / isnet-general-use / birefnet-general: rembg 패키지의 배경 제거 AI 모델 (로컬 실행, 외부 전송 없음).

모든 함수는 RGB 그림을 받아 0~255 알파(numpy uint8, HxW)를 돌려준다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

AI_METHODS = ["u2net", "isnet-general-use", "birefnet-general"]
METHODS = ["colorkey"] + AI_METHODS


def colorkey_alpha(img: Image.Image, thresh: float = 28.0, feather: float = 12.0, work: int = 512) -> np.ndarray:
    """테두리 색 기준 배경 제거 (기준선).

    thresh: 배경색과의 RGB 거리가 이 값 이하면 배경 후보
    feather: thresh부터 thresh+feather까지는 반투명으로 부드럽게
    work: 계산용으로 줄이는 긴 변 크기 (속도용)
    """
    rgb = img.convert("RGB")
    W, H = rgb.size
    k = work / max(W, H)
    small = np.asarray(rgb.resize((max(1, round(W * k)), max(1, round(H * k))), Image.BOX), dtype=np.float32)
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    bg = np.median(border, axis=0)
    dist = np.sqrt(((small - bg) ** 2).sum(-1))
    cand = dist <= thresh + feather
    # 테두리와 이어진 배경 후보만 배경으로 (그림 안쪽의 비슷한 색은 남긴다)
    reach = np.zeros_like(cand)
    reach[0, :] = cand[0, :]; reach[-1, :] = cand[-1, :]; reach[:, 0] = cand[:, 0]; reach[:, -1] = cand[:, -1]
    while True:
        grown = reach.copy()
        grown[1:, :] |= reach[:-1, :]; grown[:-1, :] |= reach[1:, :]
        grown[:, 1:] |= reach[:, :-1]; grown[:, :-1] |= reach[:, 1:]
        grown &= cand
        if (grown == reach).all():
            break
        reach = grown
    soft = np.clip((dist - thresh) / max(feather, 1e-6), 0, 1)
    alpha = np.where(reach, soft, 1.0)
    out = Image.fromarray((alpha * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    return np.asarray(out)


_sessions: dict = {}


def ai_alpha(img: Image.Image, model: str) -> np.ndarray:
    from rembg import new_session, remove

    if model not in _sessions:
        _sessions[model] = new_session(model)
    out = remove(img.convert("RGB"), session=_sessions[model], only_mask=True)
    return np.asarray(out.convert("L"))


def alpha_for(img: Image.Image, method: str, **kw) -> np.ndarray:
    if method == "colorkey":
        return colorkey_alpha(img, **kw)
    if method in AI_METHODS:
        return ai_alpha(img, method)
    raise ValueError(f"모르는 방법: {method}")


def apply_alpha(img: Image.Image, alpha: np.ndarray, crop: bool = True, pad: int = 8) -> Image.Image:
    """알파를 입혀 RGBA로 만들고, crop이면 대상 둘레만 남긴다."""
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    if crop:
        ys, xs = np.nonzero(alpha > 8)
        if len(xs):
            x0, x1 = max(0, xs.min() - pad), min(img.width, xs.max() + 1 + pad)
            y0, y1 = max(0, ys.min() - pad), min(img.height, ys.max() + 1 + pad)
            rgba = rgba.crop((x0, y0, x1, y1))
    return rgba
