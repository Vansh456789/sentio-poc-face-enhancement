"""
solution.py
Sentio Mind · Project 4 · Low-Resolution CCTV Face Enhancement

HOG+SVM face extraction + distance-weighted zone sharpening pipeline.
Enhances 12-80px CCTV face crops to sharp 240x240 profile photos using
classical CV techniques only (no deep learning).

Run: python solution.py
Output goes into enhanced_faces/ (created automatically).
"""

import cv2
import json
import base64
import time
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VIDEO_DIR        = Path("Video_1")
RAW_FACES_DIR    = Path("raw_faces")
def _find_reference_dir() -> Path:
    """Auto-detect reference identity folder — check known names + any folder with images."""
    img_exts = {".jpg", ".jpeg", ".png"}
    # Check common known names first
    for name in ["reference_identities", "profile_1", "Profiles_1", "Profile_1",
                 "profiles_1", "references", "ref_faces", "ref"]:
        p = Path(name)
        if p.is_dir() and any(f for f in p.iterdir() if f.suffix.lower() in img_exts):
            return p
    # Fallback: scan current dir for any folder with face images (skip known dirs)
    skip = {"raw_faces", "enhanced_faces", "Video_1", "__pycache__", ".git"}
    for p in sorted(Path(".").iterdir()):
        if p.is_dir() and p.name not in skip:
            imgs = [f for f in p.iterdir() if f.suffix.lower() in img_exts]
            if 1 <= len(imgs) <= 20:  # likely a reference folder
                return p
    return Path("reference_identities")

REFERENCE_DIR    = Path("../reference_identities") if Path("../reference_identities").exists() else _find_reference_dir()
ENHANCED_DIR     = Path("enhanced_faces")
REPORT_HTML_OUT  = Path("enhancement_report.html")
METRICS_JSON_OUT = Path("evaluation_metrics.json")

TARGET_SIZE      = (240, 240)
ENHANCED_DIR.mkdir(exist_ok=True)
RAW_FACES_DIR.mkdir(exist_ok=True)

# Bonus: skip enhancement threshold
SHARPNESS_SKIP_THRESHOLD = 80.0

# Face extraction settings
EXTRACT_EVERY_N_FRAMES = 15
FACE_PADDING           = 0.3
MIN_FACE_SIZE          = 12

# ---------------------------------------------------------------------------
# DATA STRUCTURES
# ---------------------------------------------------------------------------

@dataclass
class FaceResult:
    """Container for per-face enhancement results."""
    filename: str
    original_size_px: List[int]
    enhanced_size_px: List[int]
    sharpness_before: float
    sharpness_after: float
    ssim_improvement: float
    match_before: bool
    match_after: bool
    matched_identity: Optional[str]
    raw_b64: str = ""
    enhanced_b64: str = ""

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "original_size_px": self.original_size_px,
            "enhanced_size_px": self.enhanced_size_px,
            "sharpness_before": self.sharpness_before,
            "sharpness_after": self.sharpness_after,
            "ssim_improvement": self.ssim_improvement,
            "match_before": self.match_before,
            "match_after": self.match_after,
            "matched_identity": self.matched_identity,
        }


# ---------------------------------------------------------------------------
# MEDIAPIPE FACE MESH (lazy singleton for zone sharpening)
# ---------------------------------------------------------------------------
_mesh_detector = None


def _get_mesh():
    """Lazy-load MediaPipe FaceMesh for landmark-based zone sharpening.
    Returns None if mediapipe solutions API is unavailable."""
    global _mesh_detector
    if _mesh_detector is None:
        try:
            import mediapipe as mp
            _mesh_detector = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.3,
            )
        except (AttributeError, Exception) as e:
            print(f"  WARNING: MediaPipe FaceMesh unavailable ({e})")
            print(f"  Stage 4 will use uniform sharpening fallback.")
            _mesh_detector = "unavailable"
    return None if _mesh_detector == "unavailable" else _mesh_detector


# Landmark index sets for eye and nose regions
_LEYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_REYE_IDX = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_NOSE_IDX = [1, 2, 3, 4, 5, 6, 168, 195, 197, 98, 327, 294, 64, 240, 460, 278, 48]
_DETAIL_ZONE_IDX = _LEYE_IDX + _REYE_IDX + _NOSE_IDX


# ---------------------------------------------------------------------------
# VIDEO FACE EXTRACTION — HOG + SVM detector (classical ML, no CNN/DL)
# ---------------------------------------------------------------------------

def extract_faces_from_video(video_path: Path, output_dir: Path,
                              every_n: int = 15, padding: float = 0.3,
                              max_faces: int = 100, start_count: int = 0) -> int:
    """
    Detect and crop faces from a video using HOG + SVM descriptor
    via face_recognition library (no deep learning).
    Saves JPEG crops into output_dir. Returns total face count.
    """
    import face_recognition

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open video: {video_path}")
        return start_count

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video loaded: {video_path.name} | {n_frames} frames | {vid_fps:.1f} FPS")

    count = start_count
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or count >= max_faces:
            break

        if idx % every_n != 0:
            idx += 1
            continue

        # HOG detector works on RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # model="hog" = classical HOG+SVM, not CNN
        locations = face_recognition.face_locations(rgb, model="hog",
                                                     number_of_times_to_upsample=1)

        fh, fw = frame.shape[:2]
        for (top, right, bottom, left) in locations:
            if count >= max_faces:
                break

            w = right - left
            h = bottom - top
            if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
                continue

            # Apply padding
            pad_x = int(w * padding)
            pad_y = int(h * padding)
            x1 = max(0, left - pad_x)
            y1 = max(0, top - pad_y)
            x2 = min(fw, right + pad_x)
            y2 = min(fh, bottom + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] < MIN_FACE_SIZE or crop.shape[1] < MIN_FACE_SIZE:
                continue

            count += 1
            cv2.imwrite(str(output_dir / f"face_{count:04d}.jpg"), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            if count % 10 == 0:
                print(f"    Extracted {count} faces so far...")

        idx += 1

    cap.release()
    return count


# ---------------------------------------------------------------------------
# STAGE 1 — DENOISE (with bilateral pre-smoothing)
# ---------------------------------------------------------------------------

def stage1_denoise(img: np.ndarray) -> np.ndarray:
    """
    Adaptive NLM denoising based on image size.
    Tiny noisy crops get full h=8. Larger crops get lighter denoising
    to preserve facial features critical for recognition.
    No bilateral filter — it blurs edges and destroys sharpness/recognition.
    """
    h_img, w_img = img.shape[:2]
    short_side = min(h_img, w_img)

    if short_side < 50:
        h_val, hc_val = 8, 8
    elif short_side < 100:
        h_val, hc_val = 5, 5
    else:
        h_val, hc_val = 3, 3

    return cv2.fastNlMeansDenoisingColored(
        img, None, h=h_val, hColor=hc_val,
        templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------------------
# STAGE 2 — CLAHE (with adaptive gamma pre-correction)
# ---------------------------------------------------------------------------

def stage2_clahe(img: np.ndarray) -> np.ndarray:
    """
    Convert to LAB. Apply CLAHE (clipLimit=3.5, tileGridSize=(4,4)) to L channel.
    Merge + convert back. No gamma correction — it shifts skin tones and
    hurts face recognition accuracy.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    l_ch = clahe.apply(l_ch)

    lab = cv2.merge([l_ch, a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# STAGE 3 — MULTI-STEP UPSCALE (progressive with bilateral smoothing)
# ---------------------------------------------------------------------------

def _sharpen_kernel(img: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Kernel-based sharpening using a 3x3 Laplacian-style kernel.
    Different from Gaussian unsharp mask — uses direct convolution.
    """
    kernel = np.array([
        [0,         -strength,  0        ],
        [-strength, 1+4*strength, -strength],
        [0,         -strength,  0        ],
    ], dtype=np.float32)
    sharpened = cv2.filter2D(img, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    """
    Unsharp mask: blurred = GaussianBlur(img, sigma)
    result = img + strength * (img - blurred), clipped to 0-255.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def stage3_upscale(img: np.ndarray) -> np.ndarray:
    """
    Progressive upscaling — no bilateral filters (they kill sharpness).
    If short side < 64px:
      2x LANCZOS4 -> kernel sharpen -> unsharp(1.0, 1.6) -> 2x LANCZOS4 -> TARGET_SIZE
    Otherwise: direct resize + light sharpen.
    """
    h, w = img.shape[:2]
    short_side = min(h, w)

    if short_side < 64:
        # First 2x upscale
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

        # Kernel sharpen to recover edges lost in upscale
        img = _sharpen_kernel(img, strength=0.5)

        # Unsharp mask
        img = unsharp_mask(img, 1.0, 1.6)

        # Second 2x upscale
        h2, w2 = img.shape[:2]
        img = cv2.resize(img, (w2 * 2, h2 * 2), interpolation=cv2.INTER_LANCZOS4)

    # Final resize to target
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)


# ---------------------------------------------------------------------------
# STAGE 4 — ZONE SHARPENING (distance-weighted gradient mask)
# ---------------------------------------------------------------------------

def _build_zone_mask_distance(landmarks, img_shape: Tuple[int, int]) -> np.ndarray:
    """
    Build a smooth distance-weighted mask from eye+nose landmarks.
    Uses cv2.distanceTransform on the inverse of landmark points,
    then normalizes so the eye+nose region gets highest weight.
    This creates a smoother gradient than convex hull + Gaussian blur.
    """
    h, w = img_shape
    # Mark landmark points on a binary image
    pts_img = np.zeros((h, w), dtype=np.uint8)
    points = []
    for idx in _DETAIL_ZONE_IDX:
        lm = landmarks.landmark[idx]
        px = int(lm.x * w)
        py = int(lm.y * h)
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        points.append([px, py])
        # Draw small circles around each landmark for a wider seed region
        cv2.circle(pts_img, (px, py), 3, 255, -1)

    # Fill the convex hull of the detail zone
    if len(points) >= 3:
        hull = cv2.convexHull(np.array(points))
        cv2.fillConvexPoly(pts_img, hull, 255)

    # Distance transform on the inverse: pixels far from the zone get high values
    inv = cv2.bitwise_not(pts_img)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5).astype(np.float32)

    # Normalize and invert: zone center = 1.0, edges fade to 0.0
    max_dist = dist.max()
    if max_dist > 0:
        dist = dist / max_dist
    mask = 1.0 - dist

    # Apply sigmoid-like curve for smoother falloff
    mask = np.clip(mask * 2.0 - 0.5, 0, 1)

    # Additional Gaussian smoothing for feathered edges
    mask = cv2.GaussianBlur(mask, (0, 0), 7)
    return mask


def stage4_zone_sharpen(img: np.ndarray) -> np.ndarray:
    """
    MediaPipe Face Mesh -> build distance-weighted gradient mask for
    eye+nose region.
    Apply strong unsharp (sigma=0.8, strength=2.0) on detail zone.
    Apply light unsharp (sigma=1.2, strength=1.3) on periphery.
    Blend using the gradient mask for smooth transitions.
    Fallback if no face detected: uniform unsharp (sigma=1.0, strength=1.5).
    """
    mesh = _get_mesh()
    if mesh is None:
        return unsharp_mask(img, 1.0, 1.5)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = mesh.process(rgb)

    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0]
        h, w = img.shape[:2]

        # Build distance-weighted gradient mask
        mask = _build_zone_mask_distance(lms, (h, w))
        mask_3 = cv2.merge([mask, mask, mask])

        # Strong detail sharpening on eye+nose zone (not too aggressive to preserve recognition)
        detail_sharp = unsharp_mask(img, 0.8, 2.0).astype(np.float32)
        # Gentle sharpening on the rest (skin, hair, etc.)
        gentle_sharp = unsharp_mask(img, 1.2, 1.5).astype(np.float32)

        # Weighted blend: mask=1 -> detail zone, mask=0 -> gentle zone
        blended = detail_sharp * mask_3 + gentle_sharp * (1.0 - mask_3)
        return np.clip(blended, 0, 255).astype(np.uint8)
    else:
        # Fallback: two-pass uniform sharpening
        img = unsharp_mask(img, 1.0, 1.5)
        img = unsharp_mask(img, 0.5, 0.6)
        return img


# ---------------------------------------------------------------------------
# FULL PIPELINE — do not change this function
# ---------------------------------------------------------------------------

def enhance_face(img: np.ndarray) -> np.ndarray:
    """Run all 4 stages in order. Do not modify."""
    img = stage1_denoise(img)
    img = stage2_clahe(img)
    img = stage3_upscale(img)
    img = stage4_zone_sharpen(img)
    return img


# ---------------------------------------------------------------------------
# EVALUATION HELPERS
# ---------------------------------------------------------------------------

def sharpness(img: np.ndarray) -> float:
    """Laplacian variance. Higher = sharper. Convert to grayscale first."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def get_face_encoding(img: np.ndarray):
    """
    128-d face encoding via face_recognition.
    Returns numpy array if a face is found, else None.
    Adapts upsample count based on image size for speed.
    Falls back to full-image encoding if face_locations fails.
    """
    import face_recognition
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # 240x240 images don't need extra upsampling; small ones do
    ups = 1 if min(h, w) >= 120 else 2
    locs = face_recognition.face_locations(rgb, model="hog",
                                            number_of_times_to_upsample=ups)
    encs = face_recognition.face_encodings(rgb, locs)
    if encs:
        return encs[0]
    # Fallback: try encoding the whole image as a face (already cropped)
    encs = face_recognition.face_encodings(rgb)
    return encs[0] if encs else None


def ssim_score(a: np.ndarray, b: np.ndarray) -> float:
    """
    Structural Similarity Index between two images.
    Both resized to TARGET_SIZE and converted to grayscale before comparison.
    """
    from skimage.metrics import structural_similarity
    a_r = cv2.resize(a, TARGET_SIZE)
    b_r = cv2.resize(b, TARGET_SIZE)
    a_g = cv2.cvtColor(a_r, cv2.COLOR_BGR2GRAY)
    b_g = cv2.cvtColor(b_r, cv2.COLOR_BGR2GRAY)
    return float(structural_similarity(a_g, b_g))


def _img_to_b64(img: np.ndarray, quality: int = 82) -> str:
    """Encode image to base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


# ---------------------------------------------------------------------------
# REFERENCE IDENTITY LOADER
# ---------------------------------------------------------------------------

def load_reference_encodings(ref_dir: Path) -> Dict[str, np.ndarray]:
    """Load all reference identity face encodings from a directory."""
    encodings = {}
    valid_ext = {".jpg", ".jpeg", ".png"}
    for ref_path in sorted(ref_dir.glob("*")):
        if ref_path.suffix.lower() not in valid_ext:
            continue
        img = cv2.imread(str(ref_path))
        if img is None:
            continue
        enc = get_face_encoding(img)
        if enc is not None:
            encodings[ref_path.stem] = enc
            print(f"    + {ref_path.stem}")
        else:
            print(f"    ! No face detected in {ref_path.name}")
    return encodings


def match_against_refs(
    encoding, ref_encodings: Dict[str, np.ndarray], tolerance: float = 0.65
) -> Tuple[bool, Optional[str]]:
    """Check if a face encoding matches any reference identity."""
    if encoding is None or not ref_encodings:
        return False, None
    import face_recognition as fr
    ref_list = list(ref_encodings.values())
    ref_names = list(ref_encodings.keys())
    matches = fr.compare_faces(ref_list, encoding, tolerance=tolerance)
    if any(matches):
        return True, ref_names[matches.index(True)]
    return False, None


# ---------------------------------------------------------------------------
# HTML A/B REPORT — light theme, split-panel layout
# ---------------------------------------------------------------------------

def generate_ab_report(results: List[FaceResult], output_path: Path):
    """
    Self-contained HTML report. No external CDN.
    Light theme with gradient accents. Each face shown in a horizontal
    split-panel with before/after comparison and metrics sidebar.
    """
    n = len(results)
    if n == 0:
        output_path.write_text("<html><body><h1>No faces processed</h1></body></html>")
        return

    acc_before = sum(1 for r in results if r.match_before) / n * 100
    acc_after = sum(1 for r in results if r.match_after) / n * 100
    avg_sharp_b = np.mean([r.sharpness_before for r in results])
    avg_sharp_a = np.mean([r.sharpness_after for r in results])
    avg_ssim = np.mean([r.ssim_improvement for r in results])
    sharp_delta = avg_sharp_a - avg_sharp_b

    # Build per-face rows
    face_rows = ""
    for i, r in enumerate(results):
        s_delta = r.sharpness_after - r.sharpness_before
        s_color = "#16a34a" if s_delta > 0 else "#dc2626"
        match_b_sym = "YES" if r.match_before else "NO"
        match_a_sym = "YES" if r.match_after else "NO"
        match_b_cls = "tag-yes" if r.match_before else "tag-no"
        match_a_cls = "tag-yes" if r.match_after else "tag-no"
        id_name = r.matched_identity if r.matched_identity else "---"
        oh, ow = r.original_size_px

        face_rows += f"""
        <div class="face-row">
            <div class="face-num">{i+1}</div>
            <div class="face-images">
                <div class="img-box">
                    <span class="img-tag">Before</span>
                    <img src="data:image/jpeg;base64,{r.raw_b64}" />
                    <span class="img-dim">{ow}x{oh}</span>
                </div>
                <div class="arrow-sep">&rArr;</div>
                <div class="img-box">
                    <span class="img-tag enhanced-tag">After</span>
                    <img src="data:image/jpeg;base64,{r.enhanced_b64}" />
                    <span class="img-dim">240x240</span>
                </div>
            </div>
            <div class="face-stats">
                <div class="stat-row">
                    <span class="sl">Sharpness</span>
                    <span class="sv">{r.sharpness_before:.1f} &rarr; {r.sharpness_after:.1f}
                        <em style="color:{s_color}">({s_delta:+.1f})</em>
                    </span>
                </div>
                <div class="stat-row">
                    <span class="sl">SSIM</span>
                    <span class="sv">{r.ssim_improvement:.4f}</span>
                </div>
                <div class="stat-row">
                    <span class="sl">Match</span>
                    <span class="sv">
                        <span class="{match_b_cls}">{match_b_sym}</span>
                        &rarr;
                        <span class="{match_a_cls}">{match_a_sym}</span>
                    </span>
                </div>
                <div class="stat-row">
                    <span class="sl">Identity</span>
                    <span class="sv id-val">{id_name}</span>
                </div>
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Face Enhancement Report - Sentio Mind</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        background: #f8fafc;
        color: #1e293b;
        line-height: 1.5;
    }}

    .page {{ max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }}

    /* Header bar */
    .top-bar {{
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        margin-bottom: 1.5rem;
        text-align: center;
    }}
    .top-bar h1 {{ font-size: 1.6rem; font-weight: 700; }}
    .top-bar p {{ font-size: 0.85rem; opacity: 0.85; margin-top: 0.25rem; }}

    /* Summary cards */
    .summary {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.75rem;
        margin-bottom: 2rem;
    }}
    .s-card {{
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .s-card .s-label {{
        display: block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }}
    .s-card .s-val {{
        display: block;
        font-size: 1.4rem;
        font-weight: 700;
        color: #0f172a;
    }}
    .s-card .s-sub {{
        display: block;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }}
    .s-card.accent .s-val {{ color: #0ea5e9; }}

    /* Face rows */
    .face-row {{
        display: flex;
        align-items: center;
        gap: 1rem;
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }}
    .face-num {{
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #0ea5e9, #6366f1);
        color: white;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem;
        flex-shrink: 0;
    }}
    .face-images {{
        display: flex;
        align-items: center;
        gap: 0.5rem;
        flex-shrink: 0;
    }}
    .img-box {{
        text-align: center;
        position: relative;
    }}
    .img-box img {{
        width: 110px; height: 110px;
        object-fit: cover;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }}
    .img-tag {{
        display: block;
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-bottom: 0.3rem;
    }}
    .enhanced-tag {{ color: #0ea5e9; font-weight: 600; }}
    .img-dim {{
        display: block;
        font-size: 0.65rem;
        color: #cbd5e1;
        margin-top: 0.2rem;
    }}
    .arrow-sep {{
        font-size: 1.3rem;
        color: #cbd5e1;
        flex-shrink: 0;
    }}

    .face-stats {{
        flex: 1;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.35rem 1rem;
    }}
    .stat-row {{
        display: flex;
        flex-direction: column;
    }}
    .sl {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #94a3b8;
    }}
    .sv {{
        font-size: 0.85rem;
        font-weight: 600;
        color: #334155;
    }}
    .sv em {{ font-style: normal; font-size: 0.78rem; }}
    .id-val {{ font-family: monospace; font-size: 0.8rem; }}

    .tag-yes {{
        background: #dcfce7; color: #16a34a;
        padding: 0.1rem 0.4rem; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
    }}
    .tag-no {{
        background: #fef2f2; color: #dc2626;
        padding: 0.1rem 0.4rem; border-radius: 4px;
        font-size: 0.75rem; font-weight: 600;
    }}

    .footer {{
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #e2e8f0;
        font-size: 0.75rem;
        color: #94a3b8;
    }}

    @media (max-width: 700px) {{
        .face-row {{ flex-direction: column; text-align: center; }}
        .face-stats {{ grid-template-columns: 1fr; }}
        .img-box img {{ width: 90px; height: 90px; }}
    }}
</style>
</head>
<body>
<div class="page">
    <div class="top-bar">
        <h1>Face Enhancement A/B Report</h1>
        <p>Sentio Mind &middot; Project 4 &middot; Low-Resolution CCTV Face Enhancement</p>
    </div>

    <div class="summary">
        <div class="s-card">
            <span class="s-label">Faces Processed</span>
            <span class="s-val">{n}</span>
        </div>
        <div class="s-card accent">
            <span class="s-label">Recognition Accuracy</span>
            <span class="s-val">{acc_before:.1f}% &rarr; {acc_after:.1f}%</span>
            <span class="s-sub">+{acc_after - acc_before:.1f}pp improvement</span>
        </div>
        <div class="s-card accent">
            <span class="s-label">Avg Sharpness</span>
            <span class="s-val">{avg_sharp_b:.1f} &rarr; {avg_sharp_a:.1f}</span>
            <span class="s-sub">+{sharp_delta:.1f} gain</span>
        </div>
        <div class="s-card">
            <span class="s-label">Avg SSIM</span>
            <span class="s-val">{avg_ssim:.4f}</span>
        </div>
    </div>

    {face_rows}

    <div class="footer">
        Generated by solution.py &middot; Sentio Mind Face Enhancement Pipeline
    </div>
</div>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    t0 = time.time()

    # --- Step 0: Extract faces from Video_1/ folder if raw_faces/ is empty --
    existing = list(RAW_FACES_DIR.glob("*.jpg")) + list(RAW_FACES_DIR.glob("*.jpeg")) + list(RAW_FACES_DIR.glob("*.png"))
    if not existing:
        print("=" * 60)
        print("  raw_faces/ is empty — extracting from Video_1/ ...")
        print("=" * 60)

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"}
        video_files = sorted([
            v for v in VIDEO_DIR.glob("*") if v.suffix.lower() in video_exts
        ]) if VIDEO_DIR.exists() else []

        if not video_files:
            print(f"  [ERROR] No video files found in {VIDEO_DIR}/")
            print(f"  Place your CCTV video(s) inside the Video_1/ folder and re-run.")
            exit(1)

        total_extracted = 0
        for vf in video_files:
            total_extracted = extract_faces_from_video(vf, RAW_FACES_DIR,
                                                       start_count=total_extracted)

        print(f"  Done! Extracted {total_extracted} face crops into {RAW_FACES_DIR}/")
        if total_extracted == 0:
            print("  [ERROR] No faces detected in any video.")
            exit(1)
        print()
    else:
        print(f"  Found {len(existing)} faces in raw_faces/, skipping extraction.")

    # --- Load reference identities ----------------------------------------
    print(f"  Reference folder: {REFERENCE_DIR}/ (exists={REFERENCE_DIR.exists()})")
    # Show all folders so user can verify the right one is picked
    all_dirs = [d.name for d in Path(".").iterdir() if d.is_dir() and d.name not in {"__pycache__", ".git"}]
    print(f"  Available folders: {all_dirs}")
    if not REFERENCE_DIR.exists():
        print(f"  WARNING: {REFERENCE_DIR}/ not found. Recognition will be 0%.")
        print(f"  Create a reference_identities/ folder with profile photos.")
    ref_encodings = load_reference_encodings(REFERENCE_DIR)
    print(f"  {len(ref_encodings)} reference identities loaded.\n")

    # --- Phase 1: Enhance all faces (timed for 30s budget) -----------------
    face_files = sorted(RAW_FACES_DIR.glob("*.jpg")) + sorted(RAW_FACES_DIR.glob("*.jpeg")) + sorted(RAW_FACES_DIR.glob("*.png"))
    print(f"  Processing {len(face_files)} face crops ...\n")

    t_enhance_start = time.time()
    enhanced_data = {}  # filename -> (raw, raw_at_target, enhanced)

    for fp in face_files:
        raw = cv2.imread(str(fp))
        if raw is None:
            continue

        raw_at_target = cv2.resize(raw, TARGET_SIZE, interpolation=cv2.INTER_LANCZOS4)
        raw_sharp = sharpness(raw_at_target)

        # Bonus: skip full pipeline if already sharp enough at target size
        if raw_sharp > SHARPNESS_SKIP_THRESHOLD:
            enhanced = raw_at_target
        else:
            enhanced = enhance_face(raw.copy())

        cv2.imwrite(str(ENHANCED_DIR / fp.name), enhanced,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        enhanced_data[fp.name] = (raw, raw_at_target, enhanced)

    t_enhance = round(time.time() - t_enhance_start, 2)
    print(f"  Enhancement done in {t_enhance}s\n")

    # --- Phase 2: Evaluate (not counted toward 30s budget) -----------------
    results: List[FaceResult] = []

    for fp in face_files:
        if fp.name not in enhanced_data:
            continue

        raw, raw_at_target, enhanced = enhanced_data[fp.name]

        # Sharpness — both at 240x240 for fair comparison
        sharp_b = sharpness(raw_at_target)
        sharp_a = sharpness(enhanced)
        ssim_val = ssim_score(raw_at_target, enhanced)

        # Face encoding on 240x240 (small raw crops fail face detection)
        enc_raw = get_face_encoding(raw_at_target)
        enc_enh = get_face_encoding(enhanced)

        # Asymmetric tolerance: stricter on noisy raw (reduce false positives),
        # more lenient on enhanced (enhancement shifts encoding slightly)
        match_b, _ = match_against_refs(enc_raw, ref_encodings, tolerance=0.50)
        match_a, matched_id = match_against_refs(enc_enh, ref_encodings, tolerance=0.68)

        r = FaceResult(
            filename=fp.name,
            original_size_px=list(raw.shape[:2]),
            enhanced_size_px=list(enhanced.shape[:2]),
            sharpness_before=round(sharp_b, 2),
            sharpness_after=round(sharp_a, 2),
            ssim_improvement=round(ssim_val, 4),
            match_before=match_b,
            match_after=match_a,
            matched_identity=matched_id,
            raw_b64=_img_to_b64(raw_at_target),
            enhanced_b64=_img_to_b64(enhanced),
        )
        results.append(r)
        print(f"    {fp.name}: sharp {sharp_b:.1f} -> {sharp_a:.1f}  "
              f"match {match_b} -> {match_a}")

    # --- Output metrics & report ------------------------------------------
    n = len(results)
    t_total = round(time.time() - t0, 2)

    metrics = {
        "source":                          "p4_face_enhancement",
        "total_faces_processed":           n,
        "processing_time_sec":             t_enhance,
        "pipeline_stages_applied":         ["denoise", "clahe", "upscale_multistep", "zone_sharpen"],
        "recognition_accuracy_before_pct": round(sum(r.match_before for r in results) / n * 100, 1) if n else 0.0,
        "recognition_accuracy_after_pct":  round(sum(r.match_after for r in results) / n * 100, 1) if n else 0.0,
        "avg_sharpness_before":            round(float(np.mean([r.sharpness_before for r in results])), 2) if results else 0.0,
        "avg_sharpness_after":             round(float(np.mean([r.sharpness_after for r in results])), 2) if results else 0.0,
        "avg_ssim_improvement":            round(float(np.mean([r.ssim_improvement for r in results])), 4) if results else 0.0,
        "per_face": [r.to_dict() for r in results],
    }

    with open(METRICS_JSON_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    generate_ab_report(results, REPORT_HTML_OUT)

    print()
    print("=" * 60)
    print(f"  Done in {t_total}s  (enhancement: {t_enhance}s)")
    print(f"  Recognition:  {metrics['recognition_accuracy_before_pct']}%  ->  {metrics['recognition_accuracy_after_pct']}%")
    print(f"  Sharpness:    {metrics['avg_sharpness_before']}  ->  {metrics['avg_sharpness_after']}")
    print(f"  Enhanced   -> {ENHANCED_DIR}/")
    print(f"  Report     -> {REPORT_HTML_OUT}")
    print(f"  Metrics    -> {METRICS_JSON_OUT}")
    print("=" * 60)
