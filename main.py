import os
import io
import base64
import asyncio
import requests
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import fal_client

app = FastAPI(title="PhotoAdmin API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PHOTO_SIZES = {
    "cin":      {"width_mm": 35, "height_mm": 45},
    "passport": {"width_mm": 35, "height_mm": 45},
    "visa":     {"width_mm": 35, "height_mm": 45},
    "permis":   {"width_mm": 35, "height_mm": 45},
}

BG_COLORS = {
    "gray":       (210, 210, 210),
    "white":      (255, 255, 255),
    "blue":       (173, 216, 230),
    "light_blue": (200, 220, 240),
}

LAYOUTS = {
    "4x2": {"cols": 4, "rows": 2},
    "3x3": {"cols": 3, "rows": 3},
    "2x2": {"cols": 2, "rows": 2},
    "1x1": {"cols": 1, "rows": 1},
}

FACE_RATIO     = 0.75
HEADROOM_RATIO = 0.08


def mm_to_px(mm, dpi):
    return int(mm / 25.4 * dpi)


def detect_face(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    return faces[0]


def face_aware_crop(img, target_w, target_h, zoom=1.0):
    img_rgb = np.array(img.convert("RGB"))
    ih, iw = img_rgb.shape[:2]
    face = detect_face(img_rgb)

    if face is not None:
        fx, fy, fw, fh = face
        face_cx = fx + fw // 2
        crop_h = int((fh / FACE_RATIO) * zoom)
        crop_w = int(crop_h * target_w / target_h)
        headroom_px = int(crop_h * HEADROOM_RATIO)
        crop_top  = fy - headroom_px
        crop_left = face_cx - crop_w // 2

        pad_top    = max(0, -crop_top)
        pad_left   = max(0, -crop_left)
        pad_bottom = max(0, (crop_top + crop_h) - ih)
        pad_right  = max(0, (crop_left + crop_w) - iw)

        if any([pad_top, pad_left, pad_bottom, pad_right]):
            img_rgb = np.pad(img_rgb, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='edge')
            crop_top  += pad_top
            crop_left += pad_left

        cropped = img_rgb[crop_top:crop_top+crop_h, crop_left:crop_left+crop_w]
    else:
        target_ratio = target_w / target_h
        src_ratio = iw / ih
        if src_ratio > target_ratio:
            crop_w = int(ih * target_ratio)
            crop_left = (iw - crop_w) // 2
            cropped = img_rgb[0:ih, crop_left:crop_left+crop_w]
        else:
            crop_h = int(iw / target_ratio)
            top_offset = int(ih * 0.02)
            crop_h = min(crop_h, ih - top_offset)
            cropped = img_rgb[top_offset:top_offset+crop_h, 0:iw]

    result = cv2.resize(cropped, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    return Image.fromarray(result)


def build_sheet(photo, cols, rows, pad):
    W = photo.width * cols + pad * (cols + 1)
    H = photo.height * rows + pad * (rows + 1)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            sheet.paste(photo, (pad + c * (photo.width + pad), pad + r * (photo.height + pad)))
    return sheet


async def fal_remove_bg(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    try:
        result = await asyncio.to_thread(
            fal_client.subscribe,
            "fal-ai/birefnet/v2",
            arguments={"image_url": data_uri, "model": "Portrait", "operating_resolution": "1024x1024"},
        )
        resp = requests.get(result["image"]["url"], timeout=30)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(500, f"خطأ في إزالة الخلفية: {str(e)}")


async def fal_upscale(image_bytes):
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    PROMPT = (
        "Ultra-high-resolution 4K enhancement based strictly on the provided reference image. "
        "Absolute fidelity to original facial anatomy, proportions, and identity. "
        "Preserve expression, gaze, pose, camera angle, framing, and perspective with zero deviation. "
        "Clothing, hair, skin, and background elements must remain unchanged. "
        "Recover fine-grain detail with natural realism. No smoothing, no plastic skin, no artificial gloss."
    )
    try:
        result = await asyncio.to_thread(
            fal_client.subscribe,
            "fal-ai/clarity-upscaler",
            arguments={"image_url": data_uri, "prompt": PROMPT, "upscale_factor": 2, "creativity": 0, "resemblance": 1.0},
        )
        resp = requests.get(result["image"]["url"], timeout=60)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return image_bytes


@app.get("/")
async def root():
    return {"status": "ok", "service": "PhotoAdmin API", "version": "2.0.0"}


@app.get("/api/health")
async def health():
    return {"status": "ok", "fal_configured": bool(os.getenv("FAL_KEY")), "version": "2.0.0"}


@app.post("/api/biometric-photo")
async def biometric_photo(
    file:     UploadFile = File(...),
    doc_type: str   = Form("cin"),
    bg_color: str   = Form("gray"),
    layout:   str   = Form("4x2"),
    dpi:      int   = Form(300),
    zoom:     float = Form(1.0),
    upscale:  bool  = Form(False),
):
    if doc_type not in PHOTO_SIZES: raise HTTPException(400, "doc_type غير مدعوم")
    if bg_color not in BG_COLORS:   raise HTTPException(400, "bg_color غير مدعوم")
    if layout   not in LAYOUTS:     raise HTTPException(400, "layout غير مدعوم")
    if dpi not in (150, 300, 600):  raise HTTPException(400, "dpi غير مدعوم")

    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024: raise HTTPException(400, "حجم الصورة أكبر من 15MB")

    # 1. إزالة الخلفية
    cutout = await fal_remove_bg(raw)

    # 2. إضافة الخلفية
    bg = Image.new("RGBA", cutout.size, (*BG_COLORS[bg_color], 255))
    final = Image.alpha_composite(bg, cutout).convert("RGB")

    # 3. القص الذكي
    size = PHOTO_SIZES[doc_type]
    tw = mm_to_px(size["width_mm"], dpi)
    th = mm_to_px(size["height_mm"], dpi)
    photo = face_aware_crop(final, tw, th, zoom=zoom)

    # 4. Upscale اختياري
    if upscale:
        buf_tmp = io.BytesIO()
        photo.save(buf_tmp, format="JPEG", quality=95)
        upscaled = await fal_upscale(buf_tmp.getvalue())
        photo = Image.open(io.BytesIO(upscaled)).convert("RGB").resize((tw, th), Image.LANCZOS)

    # 5. لوحة الصور
    lyt = LAYOUTS[layout]
    sheet = build_sheet(photo, lyt["cols"], lyt["rows"], mm_to_px(3, dpi))

    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=97, dpi=(dpi, dpi))

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="photo_{doc_type}_{layout}.jpg"'},
    )


@app.post("/api/biometric-photo/preview")
async def biometric_preview(
    file:     UploadFile = File(...),
    doc_type: str   = Form("cin"),
    bg_color: str   = Form("gray"),
    zoom:     float = Form(1.0),
):
    raw = await file.read()
    cutout = await fal_remove_bg(raw)
    bg = Image.new("RGBA", cutout.size, (*BG_COLORS[bg_color], 255))
    final = Image.alpha_composite(bg, cutout).convert("RGB")
    size = PHOTO_SIZES[doc_type]
    pw = mm_to_px(size["width_mm"], 150)
    ph = mm_to_px(size["height_mm"], 150)
    photo = face_aware_crop(final, pw, ph, zoom=zoom)
    buf = io.BytesIO()
    photo.save(buf, format="JPEG", quality=85)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
# ═══════════════════════════════════════════════════════════════
#  أضف هذا الكود في نهاية ملف main.py الموجود على Railway
# ═══════════════════════════════════════════════════════════════

# في أعلى الملف (مع الـ imports الموجودة) أضف:
# import qrcode
# from family_card_api import generate_family_card   ← إذا وضعت الكود في ملف منفصل
# أو انسخ محتوى family_card_api.py مباشرة في main.py

@app.post("/api/family-card")
async def family_card_endpoint(
    photo:               UploadFile = File(...),
    husband_name_ar:     str = Form(""),
    husband_name_fr:     str = Form(""),
    husband_cnie:        str = Form(""),
    husband_birth_date:  str = Form(""),
    husband_birth_place: str = Form(""),
    husband_reg_num:     str = Form(""),
    wife_name_ar:        str = Form(""),
    wife_name_fr:        str = Form(""),
    wife_cnie:           str = Form(""),
    wife_birth_date:     str = Form(""),
    wife_birth_place:    str = Form(""),
    wife_reg_num:        str = Form(""),
    phone:               str = Form(""),
    address_ar:          str = Form(""),
    address_fr:          str = Form(""),
    reg_num_1:           str = Form(""),
    reg_num_2:           str = Form(""),
    card_ref:            str = Form(""),
    google_drive_url:    str = Form(""),
):
    """توليد البطاقة العائلية كصورة JPG"""

    SVG_TEMPLATE = os.path.join(os.path.dirname(__file__), "family_card_template.svg")
    if not os.path.exists(SVG_TEMPLATE):
        raise HTTPException(500, "ملف القالب غير موجود على السيرفر")

    jpg_bytes = await generate_family_card(
        photo=photo,
        husband_name_ar=husband_name_ar,
        husband_name_fr=husband_name_fr,
        husband_cnie=husband_cnie,
        husband_birth_date=husband_birth_date,
        husband_birth_place=husband_birth_place,
        husband_reg_num=husband_reg_num,
        wife_name_ar=wife_name_ar,
        wife_name_fr=wife_name_fr,
        wife_cnie=wife_cnie,
        wife_birth_date=wife_birth_date,
        wife_birth_place=wife_birth_place,
        wife_reg_num=wife_reg_num,
        phone=phone,
        address_ar=address_ar,
        address_fr=address_fr,
        reg_num_1=reg_num_1,
        reg_num_2=reg_num_2,
        card_ref=card_ref,
        google_drive_url=google_drive_url,
        svg_template_path=SVG_TEMPLATE,
    )

    return Response(
        content=jpg_bytes,
        media_type="image/jpeg",
        headers={"Content-Disposition": 'attachment; filename="family_card.jpg"'},
    )
