"""Image forensics for document tampering and low-quality / recaptured images.

Every analyser returns numeric features (used by the ML model) and the
pipeline turns those into human readable findings.
"""
from __future__ import annotations

import io
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ExifTags

from .findings import Finding

EDITING_SOFTWARE = ("photoshop", "gimp", "canva", "paint", "pixlr", "snapseed",
                    "picsart", "affinity", "lightroom", "illustrator", "inkscape",
                    "coreldraw", "fotor", "photopea")


def load_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


# ---------------------------------------------------------------- metadata
def metadata_features(img: Image.Image) -> Tuple[Dict, List[Finding]]:
    findings: List[Finding] = []
    exif = {}
    try:
        raw = img.getexif()
        exif = {ExifTags.TAGS.get(k, k): v for k, v in raw.items()}
    except Exception:
        pass
    software = str(exif.get("Software", "")).lower()
    edited = any(s in software for s in EDITING_SOFTWARE)
    has_exif = bool(exif)
    feats = {"exif_present": float(has_exif), "edited_software": float(edited)}
    if edited:
        findings.append(Finding("forensics", "EDITING_SOFTWARE", "high",
                                f"Image metadata shows it was processed with '{exif.get('Software')}'.",
                                {"software": str(exif.get("Software"))}))
    if "DateTime" in exif and "DateTimeOriginal" in exif and exif["DateTime"] != exif["DateTimeOriginal"]:
        findings.append(Finding("forensics", "EXIF_TIME_MISMATCH", "medium",
                                "Modification time differs from capture time, which suggests post-editing.",
                                {"modified": str(exif["DateTime"]), "original": str(exif["DateTimeOriginal"])}))
        feats["edited_software"] = max(feats["edited_software"], 0.5)
    if not has_exif:
        findings.append(Finding("forensics", "NO_EXIF", "info",
                                "No EXIF metadata (normal for scans and screenshots; stripped by many editors).", {}))
    return feats, findings


# ----------------------------------------------------------------- quality
def quality_features(gray: np.ndarray, img: Image.Image) -> Tuple[Dict, List[Finding]]:
    findings: List[Finding] = []
    h, w = gray.shape
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    feats = {"blur_var": blur, "megapixels": h * w / 1e6}
    if blur < 40:
        findings.append(Finding("forensics", "LOW_SHARPNESS", "low",
                                "Image is very blurry, so text and security features cannot be verified.",
                                {"laplacian_variance": round(blur, 1)}))
    if min(h, w) < 400:
        findings.append(Finding("forensics", "LOW_RESOLUTION", "low",
                                "Image resolution is too low for reliable verification.", {"size": [w, h]}))
    return feats, findings


# --------------------------------------------------- Error Level Analysis
def ela_features(img: Image.Image, quality: int = 90, block: int = 32) -> Tuple[Dict, List[Finding], np.ndarray]:
    """Re-compress and compare. Regions that were edited or pasted respond
    differently from their surroundings.

    Raw ELA is dominated by edges (text always scores high), so each block's
    ELA is first explained by its local edge energy with a robust linear fit;
    only blocks that deviate from that fit - high *or* low - count as anomalies.
    """
    rgb = img.convert("RGB")
    buf = io.BytesIO()
    rgb.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = np.abs(np.asarray(rgb, dtype=np.float32) - np.asarray(resaved, dtype=np.float32)).mean(axis=2)
    g = np.asarray(rgb.convert("L"), dtype=np.float32)
    gy, gx = np.gradient(cv2.GaussianBlur(g, (0, 0), 1.0))
    edge = np.abs(gx) + np.abs(gy)
    h, w = diff.shape
    bh, bw = h // block, w // block
    base = {"ela_mean": float(diff.mean())}
    if bh < 3 or bw < 3:
        return {**base, "ela_outlier_ratio": 0.0, "ela_max_z": 0.0}, [], diff
    cut = (slice(0, bh * block), slice(0, bw * block))
    e = diff[cut].reshape(bh, block, bw, block).mean(axis=(1, 3)).ravel()
    gr = edge[cut].reshape(bh, block, bw, block).mean(axis=(1, 3)).ravel()
    # robust fit e ~ a + b*gr (iteratively re-weighted)
    A = np.stack([np.ones_like(gr), gr], axis=1)
    wts = np.ones_like(e)
    for _ in range(5):
        coef, *_ = np.linalg.lstsq(A * wts[:, None], e * wts, rcond=None)
        res = e - A @ coef
        mad = np.median(np.abs(res - np.median(res))) * 1.4826 + 0.02
        wts = 1.0 / np.maximum(1.0, np.abs(res) / (2.5 * mad))
    z = (res - np.median(res)) / mad
    outliers = float((np.abs(z) > 7).mean())
    feats = {**base, "ela_outlier_ratio": outliers, "ela_max_z": float(np.abs(z).max())}
    findings: List[Finding] = []
    if outliers > 0.05 and np.abs(z).max() > 12:
        sev = "medium" if outliers > 0.12 else "low"
        findings.append(Finding("forensics", "ELA_ANOMALY", sev,
                                "Error-level analysis found regions whose compression behaviour does not fit the rest of the image, a typical sign of pasted or edited content.",
                                {"outlier_block_ratio": round(outliers, 4), "max_z": round(float(np.abs(z).max()), 1)}))
    return feats, findings, diff


# ------------------------------------------------ Noise inconsistency
def noise_features(gray: np.ndarray, block: int = 48) -> Tuple[Dict, List[Finding]]:
    """Camera/scanner noise is fairly uniform; spliced regions carry different noise."""
    blurred = cv2.GaussianBlur(gray, (0, 0), 1.5)
    resid = gray.astype(np.float32) - blurred.astype(np.float32)
    h, w = resid.shape
    bh, bw = h // block, w // block
    if bh < 3 or bw < 3:
        return {"noise_cv": 0.0, "noise_outlier_ratio": 0.0}, []
    cells = resid[: bh * block, : bw * block].reshape(bh, block, bw, block).std(axis=(1, 3))
    # Ignore near-flat background blocks (std≈0) which naturally have little noise.
    active = cells[cells > 0.4]
    if active.size < 6:
        return {"noise_cv": 0.0, "noise_outlier_ratio": 0.0}, []
    med = float(np.median(active))
    mad = float(np.median(np.abs(active - med))) + 1e-6
    z = np.abs(active - med) / (1.4826 * mad)
    ratio = float((z > 5).mean())
    cv = float(active.std() / (active.mean() + 1e-6))
    feats = {"noise_cv": cv, "noise_outlier_ratio": ratio}
    findings: List[Finding] = []
    if ratio > 0.06 and cv > 0.6:
        findings.append(Finding("forensics", "NOISE_INCONSISTENT", "medium",
                                "Noise level varies sharply between regions, which can indicate spliced or re-rendered areas.",
                                {"outlier_ratio": round(ratio, 3), "cv": round(cv, 2)}))
    return feats, findings


# ---------------------------------------------------- Copy-move detection
def _candidate_shifts(gray: np.ndarray, min_dist: int = 40, top: int = 3):
    orb = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
    kps, des = orb.detectAndCompute(gray, None)
    if des is None or len(kps) < 30:
        return []
    knn = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des, des, k=3)
    buckets: Dict[Tuple[int, int], List[np.ndarray]] = {}
    for m in knn:
        if len(m) < 3:
            continue
        c, n = m[1], m[2]            # m[0] is the keypoint matched with itself
        if c.distance < 0.6 * n.distance and c.distance < 40:
            p, q = np.array(kps[c.queryIdx].pt), np.array(kps[c.trainIdx].pt)
            if np.linalg.norm(p - q) > min_dist:
                d = q - p
                if d[0] < 0 or (d[0] == 0 and d[1] < 0):
                    d = -d                     # direction-independent
                buckets.setdefault((int(round(d[0] / 6)), int(round(d[1] / 6))), []).append(d)
    ranked = sorted(buckets.values(), key=len, reverse=True)[:top]
    return [(np.median(np.array(b), axis=0), len(b)) for b in ranked if len(b) >= 4]


def _noise_duplication(gray: np.ndarray, shift: np.ndarray, block: int = 24, search: int = 3) -> Tuple[int, int]:
    """Count flat blocks whose *sensor-noise residual* is the same at p and p+shift.
    Glyph repeats have different noise; true clones carry the copied noise with them.
    Vectorised: local normalised cross-correlation via box filters."""
    g32 = gray.astype(np.float32)
    R = g32 - cv2.GaussianBlur(g32, (0, 0), 1.2)
    gy, gx = np.gradient(cv2.GaussianBlur(g32, (0, 0), 2.0))
    flat = ((np.abs(gx) + np.abs(gy)) < 1.2).astype(np.uint8)
    flat_block = cv2.erode(flat, np.ones((block, block), np.uint8)).astype(np.float32)
    k = (block, block)
    mu = cv2.blur(R, k)
    var = cv2.blur(R * R, k) - mu * mu
    h, w = R.shape
    ones = np.ones_like(R)
    best_hits, best_total = 0, 0
    bx0, by0 = int(round(shift[0])), int(round(shift[1]))
    for dy in range(-search, search + 1):
        for dx in range(-search, search + 1):
            M = np.float32([[1, 0, -(bx0 + dx)], [0, 1, -(by0 + dy)]])
            warp = lambda m: cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            R2, mu2, var2, fb2, valid = warp(R), warp(mu), warp(var), warp(flat_block), warp(ones)
            cov = cv2.blur(R * R2, k) - mu * mu2
            ncc = cov / np.sqrt(np.maximum(var * var2, 1e-6))
            ok = (flat_block > 0.5) & (fb2 > 0.5) & (valid > 0.5) & (var > 0.36) & (var2 > 0.36)
            ok_s = ok[block // 2::block, block // 2::block]
            total = int(ok_s.sum())
            hits = int((ok_s & (ncc[block // 2::block, block // 2::block] > 0.6)).sum())
            if hits > best_hits:
                best_hits, best_total = hits, total
    return best_hits, best_total


def copy_move_features(gray: np.ndarray) -> Tuple[Dict, List[Finding]]:
    """Detect duplicated regions (cloned text, stamps, signatures).

    Step 1: ORB keypoints propose candidate translation vectors.
    Step 2: each vector is verified by checking that the flat-area sensor noise
            is duplicated, which separates real clones from repeated glyphs.
    """
    if max(gray.shape) > 1400:
        gray = cv2.resize(gray, None, fx=1400 / max(gray.shape), fy=1400 / max(gray.shape), interpolation=cv2.INTER_AREA)
    feats = {"copy_move_matches": 0.0, "copy_move_consistent": 0.0}
    findings: List[Finding] = []
    best = (0, 0, 0)
    for shift, n in _candidate_shifts(gray):
        hits, total = _noise_duplication(gray, shift)
        if hits > best[0]:
            best = (hits, total, n)
    feats["copy_move_matches"] = float(best[2])
    feats["copy_move_consistent"] = float(best[0])
    if best[0] >= 4:
        findings.append(Finding("forensics", "COPY_MOVE", "high",
                                "A region of the image is duplicated, including its sensor noise, which indicates cloned content such as copied text, stamps or signatures.",
                                {"duplicated_noise_blocks": best[0], "keypoint_pairs": best[2]}))
    return feats, findings


# ---------------------------------------------------------------- faces
_FACE = None


def face_features(gray: np.ndarray) -> Tuple[Dict, List[Finding], int]:
    """Informational only. OpenCV 5 dropped the Haar cascade API, so this
    degrades gracefully (face_count = -1) when it is unavailable."""
    global _FACE
    try:
        if _FACE is None:
            _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = _FACE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        n = len(faces)
    except Exception:
        n = -1
    return {"face_count": float(n)}, [], n


# --------------------------------------------------------- perceptual hash
def dhash(img: Image.Image, size: int = 16) -> str:
    g = np.asarray(img.convert("L").resize((size + 1, size), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else len(a)


# ----------------------------------------------------------- orchestrator
def analyse_image(data: bytes) -> Tuple[Dict, List[Finding], Dict]:
    img = load_image(data)
    gray = np.asarray(img.convert("L"))
    feats: Dict[str, float] = {}
    findings: List[Finding] = []
    extra: Dict = {"size": list(img.size), "format": img.format}
    for fn in (lambda: metadata_features(img), lambda: quality_features(gray, img),
               lambda: noise_features(gray), lambda: copy_move_features(gray)):
        f, fi = fn()
        feats.update(f); findings += fi
    f, fi, _ = ela_features(img)
    feats.update(f); findings += fi
    f, _, n_faces = face_features(gray)
    feats.update(f)
    extra["faces"] = n_faces
    extra["dhash"] = dhash(img)
    return feats, findings, extra
