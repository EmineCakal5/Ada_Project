# -*- coding: utf-8 -*-
"""OpenCV yerine Pillow ile BGR frame üzerinde Unicode/Türkçe metin."""

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

_FONT_CACHE: Dict[Tuple[Optional[str], int], object] = {}


def _resolve_font_path() -> Optional[str]:
    if os.name == "nt":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for name in ("segoeui.ttf", "arial.ttf", "calibri.ttf"):
            p = os.path.join(windir, "Fonts", name)
            if os.path.isfile(p):
                return p
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if os.path.isfile(p):
            return p
    return None


def _load_font(size_px: int):
    from PIL import ImageFont
    path = _resolve_font_path()
    key = (path, size_px)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    try:
        font = ImageFont.truetype(path, size_px) if path else ImageFont.load_default()
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _text_size(text: str, font_px: int) -> Tuple[int, int]:
    from PIL import Image, ImageDraw
    if not text:
        return 0, 0
    img = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(img)
    font = _load_font(font_px)
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return max(1, r - l), max(1, b - t)


def draw_text_bgr(
    frame: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_px: int,
    color_bgr: Tuple[int, int, int],
    stroke_width: int = 0,
    stroke_color: Tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """BGR numpy görüntü üzerinde metin; org = sol-alt taban (OpenCV putText uyumu)."""
    from PIL import Image, ImageDraw
    if text is None or frame is None:
        return frame
    s = str(text)
    if not s:
        return frame
    x, y = int(org[0]), int(org[1])
    rgb = (int(color_bgr[2]), int(color_bgr[1]), int(color_bgr[0]))
    stroke_rgb = (int(stroke_color[2]), int(stroke_color[1]), int(stroke_color[0]))
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    font = _load_font(font_px)
    draw.text(
        (x, y), s, font=font, fill=rgb, anchor="ls",
        stroke_width=stroke_width or 0,
        stroke_fill=stroke_rgb if stroke_width else None,
    )
    out = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    frame[:] = out
    return frame
