"""Synthetic *demo* ID cards and tampering simulators.

The cards are deliberately generic ("DEMO IDENTITY CARD", no government
branding). They exist only to test the detector and to train the demo model.
"""
from __future__ import annotations

import io
import random
from typing import Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .validators import verhoeff_generate_check_digit

_FONT_PATHS = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
               "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"]


def _font(size: int):
    for p in _FONT_PATHS:
        try:
            return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def valid_aadhaar(rng: random.Random) -> str:
    base = str(rng.randint(2, 9)) + "".join(str(rng.randint(0, 9)) for _ in range(10))
    return base + verhoeff_generate_check_digit(base)


def valid_pan(rng: random.Random, surname: str) -> str:
    L = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    return "".join(rng.choice(L) for _ in range(3)) + "P" + surname[0].upper() + \
        f"{rng.randint(1, 9999):04d}" + rng.choice(L)


def make_card(name: str, dob: str, aadhaar: str, pan: str, seed: int = 0, size=(1000, 630)) -> Image.Image:
    rng = np.random.RandomState(seed)
    w, h = size
    img = Image.new("RGB", size, (236, 241, 247))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 90], fill=(30, 64, 120))
    d.text((30, 25), "DEMO IDENTITY CARD  -  SAMPLE ONLY", font=_font(36), fill="white")
    d.rectangle([40, 140, 280, 420], fill=(190, 200, 215), outline=(90, 100, 120), width=3)
    d.ellipse([105, 165, 215, 275], fill=(150, 160, 175))
    d.ellipse([75, 285, 245, 460], fill=(150, 160, 175))
    f, fb = _font(34), _font(38)
    y = 150
    for label, val in (("Name", name), ("DOB", dob), ("Aadhaar", " ".join([aadhaar[i:i + 4] for i in range(0, 12, 4)])), ("PAN", pan)):
        d.text((320, y), f"{label}:", font=f, fill=(70, 80, 100))
        d.text((520, y), val, font=fb, fill=(10, 10, 20))
        y += 85
    d.rectangle([0, h - 60, w, h], fill=(30, 64, 120))
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.normal(0, 3.2, arr.shape)           # uniform sensor noise
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.6))


def to_jpeg(img: Image.Image, quality=85, software: str | None = None) -> bytes:
    buf = io.BytesIO()
    kw = {}
    if software:
        ex = Image.Exif()
        ex[0x0131] = software
        kw["exif"] = ex.tobytes()
    img.save(buf, "JPEG", quality=quality, **kw)
    return buf.getvalue()


# ------------------------------------------------------------ tampering
def tamper_text(img: Image.Image, new_dob: str, seed: int = 1) -> Image.Image:
    """Overwrite the DOB field with a clean, noise-free patch (typical edit)."""
    img = img.copy()
    d = ImageDraw.Draw(img)
    box = [505, 220, 940, 285]
    d.rectangle(box, fill=(236, 241, 247))
    d.text((520, 235), new_dob, font=_font(38), fill=(10, 10, 20))
    return img


def tamper_splice(img: Image.Image, seed: int = 2) -> Image.Image:
    """Paste a patch that came from a differently-processed source."""
    rng = np.random.RandomState(seed)
    img = img.copy()
    patch = Image.new("RGB", (430, 70), (236, 241, 247))
    ImageDraw.Draw(patch).text((15, 12), "01/01/1990", font=_font(38), fill=(10, 10, 20))
    patch = patch.filter(ImageFilter.GaussianBlur(1.2))
    buf = io.BytesIO(); patch.save(buf, "JPEG", quality=55); buf.seek(0)
    img.paste(Image.open(buf), (505, 222))
    return img


def tamper_clone(img: Image.Image) -> Image.Image:
    """Clone-stamp: copy the name row (text + background + noise) lower on the card."""
    img = img.copy()
    region = img.crop((300, 140, 960, 235))
    img.paste(region, (300, 450))
    return img
