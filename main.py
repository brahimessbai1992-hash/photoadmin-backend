import os
import io
import base64
import asyncio
import requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import fal_client

app = FastAPI(title="PhotoAdmin API", version="1.0.0")

# ── CORS: السماح لـ Vercel و localhost ──────────────────────────────────────
ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "https://*.vercel.app",
    os.getenv("FRONTEND_URL", ""),
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # سنضيق هذا بعد رفع الـ domain النهائي
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── إعدادات الأبعاد والألوان ────────────────────────────────────────────────
PHOTO_SIZES = {
    "cin":      {"width_mm": 35, "height_mm": 45},
    "passport": {"width_mm": 35, "height_mm": 45},
    "visa":     {"width_mm": 35, "height_mm": 45},
    "permis":   {"width_mm": 35, "height_mm": 45},
}

BG_COLORS = {
    "gray":  (210, 210, 210),
    "white": (255, 255, 255),
    "blue":  (173, 216, 230),
}

LAYOUTS = {
    "4x2": {"cols": 4, "rows": 2},
    "3x3": {"cols": 3, "rows": 3},
    "2x2": {"cols": 2, "rows": 2},
}


def mm_to_px(mm: float, dpi: int) -> int:
    return int(mm / 25.4 * dpi)


def smart_crop(img: Image.Image, w: int, h: int) -> Image.Image:
    sw, sh = img.size
    ratio = w / h
    if (sw / sh) > ratio:
        nw = int(sh * ratio)
        img = img.crop(((sw - nw) // 2, 0, (sw - nw) // 2 + nw, sh))
    else:
        nh = int(sw / ratio)
        img = img.crop((0, 0, sw, nh))
    return img.resize((w, h), Image.LANCZOS)


def build_sheet(photo: Image.Image, cols: int, rows: int, pad: int) -> Image.Image:
    W = photo.width * cols + pad * (cols + 1)
    H = photo.height * rows + pad * (rows + 1)
    sheet = Image.new("RGB", (W, H), (255, 255, 255))
    for r in range(rows):
        for c in range(cols):
            sheet.paste(photo, (pad + c * (photo.width + pad), pad + r * (photo.height + pad)))
    return sheet


async def fal_remove_bg(image_bytes: bytes) -> Image.Image:
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
        raise HTTPException(500, f"Fal.ai error: {str(e)}")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"status": "ok", "service": "PhotoAdmin API"}


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "fal_configured": bool(os.getenv("FAL_KEY")),
        "docs": list(PHOTO_SIZES.keys()),
    }


@app.post("/api/biometric-photo")
async def biometric_photo(
    file: UploadFile = File(...),
    doc_type: str = Form("cin"),
    bg_color: str = Form("gray"),
    layout:   str = Form("4x2"),
    dpi:      int = Form(300),
):
    if doc_type not in PHOTO_SIZES: raise HTTPException(400, "doc_type غير مدعوم")
    if bg_color not in BG_COLORS:   raise HTTPException(400, "bg_color غير مدعوم")
    if layout   not in LAYOUTS:     raise HTTPException(400, "layout غير مدعوم")
    if dpi not in (150, 300, 600):  raise HTTPException(400, "dpi غير مدعوم")

    raw = await file.read()
    if len(raw) > 10 * 1024 * 1024: raise HTTPException(400, "حجم الصورة أكبر من 10MB")

    # 1. إزالة الخلفية
    cutout = await fal_remove_bg(raw)

    # 2. إضافة خلفية ملونة
    bg = Image.new("RGBA", cutout.size, (*BG_COLORS[bg_color], 255))
    final = Image.alpha_composite(bg, cutout).convert("RGB")

    # 3. قص وتحجيم
    size = PHOTO_SIZES[doc_type]
    w = mm_to_px(size["width_mm"], dpi)
    h = mm_to_px(size["height_mm"], dpi)
    photo = smart_crop(final, w, h)

    # 4. لوحة الصور
    lyt = LAYOUTS[layout]
    sheet = build_sheet(photo, lyt["cols"], lyt["rows"], mm_to_px(3, dpi))

    # 5. تصدير
    buf = io.BytesIO()
    sheet.save(buf, format="JPEG", quality=95, dpi=(dpi, dpi))

    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"Content-Disposition": f'attachment; filename="photo_{doc_type}_{layout}.jpg"'},
    )
