"""
=============================================================
  البطاقة العائلية — Family Card Generator
  يُضاف هذا الكود إلى main.py الخاص بـ Railway
=============================================================
"""

import os, io, base64, asyncio, re, requests
import cv2, numpy as np, qrcode
from PIL import Image, ImageDraw, ImageFont
from fastapi import UploadFile, File, Form, HTTPException
from fastapi.responses import Response
import fal_client

# ── إعدادات البطاقة ──────────────────────────────────────────────────────────
CARD_W       = 2437    # بكسل (243.78mm × 10)
CARD_H       = 1530    # بكسل (153.07mm × 10)
SCALE        = 10      # نسبة التكبير من وحدات SVG إلى بكسل

# منطقة الصورة الشخصية (من SVG: x=8.86 y=34.07 w=51.15 h=51.15)
PHOTO_X, PHOTO_Y = 88,  340
PHOTO_W, PHOTO_H = 511, 511
PHOTO_RADIUS     = 120  # انحناء الزوايا (rx=12 في SVG)

# منطقة QR Code (من SVG: x=23.24 y=96.32 w=22.68 h=22.68)
QR_X, QR_Y = 232, 963
QR_W, QR_H = 226, 226

# ── مواضع حقول النص (x, y, محاذاة) ─────────────────────────────────────────
# المحاذاة: 'r' = يمين (عربي)، 'l' = يسار (فرنسي/أرقام)
TEXT_FIELDS = {
    "husband_name_ar":    (1646, 309,  'r', 32, 'bold'),
    "husband_name_fr":    (1090, 351,  'l', 26, 'normal'),
    "husband_cnie":       (1081, 421,  'l', 26, 'normal'),
    "husband_birth_date": (1272, 549,  'l', 26, 'normal'),
    "husband_birth_place":(1376, 619,  'r', 24, 'normal'),
    "husband_reg_num":    (1233, 689,  'l', 26, 'normal'),
    "wife_name_ar":       (1589, 789,  'r', 32, 'bold'),
    "wife_name_fr":       (1100, 831,  'l', 26, 'normal'),
    "wife_cnie":          (1079, 901,  'l', 26, 'normal'),
    "wife_birth_date":    (1271, 1029, 'l', 26, 'normal'),
    "wife_birth_place":   (1376, 1099, 'r', 24, 'normal'),
    "wife_reg_num":       (1229, 1169, 'l', 26, 'normal'),
    "phone":              (1799, 1279, 'l', 24, 'normal'),
    "address_ar":         (1553, 1436, 'r', 22, 'normal'),
    "address_fr":         (101,  1436, 'l', 22, 'normal'),
    "reg_num_1":          (989,  1281, 'l', 24, 'normal'),
    "reg_num_2":          (984,  1351, 'l', 24, 'normal'),
    "card_ref":           (130,  1341, 'l', 22, 'normal'),
}

# ── تحميل خط عربي ────────────────────────────────────────────────────────────
def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """يحاول تحميل خط عربي، يرجع للافتراضي عند الفشل"""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf" if bold
            else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf" if bold
            else "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def reshape_arabic(text: str) -> str:
    """إعادة تشكيل النص العربي للعرض الصحيح"""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except ImportError:
        return text


def draw_text_field(draw: ImageDraw.Draw, text: str, x: int, y: int,
                    align: str, size: int, weight: str, color=(30, 30, 30)):
    """رسم حقل نص مع دعم العربية"""
    if not text or not text.strip():
        return
    font = get_font(size, bold=(weight == 'bold'))
    display_text = reshape_arabic(text) if align == 'r' else text
    bbox = draw.textbbox((0, 0), display_text, font=font)
    text_w = bbox[2] - bbox[0]
    draw_x = x - text_w if align == 'r' else x
    draw.text((draw_x, y - size), display_text, font=font, fill=color)


# ── تحويل SVG إلى صورة خلفية ─────────────────────────────────────────────────
def svg_to_background(svg_path: str) -> Image.Image:
    """
    تحويل SVG إلى PNG عبر cairosvg (على السيرفر)
    أو استخدام Inkscape كبديل
    """
    try:
        import cairosvg
        png_bytes = cairosvg.svg2png(
            url=svg_path,
            output_width=CARD_W,
            output_height=CARD_H
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except ImportError:
        pass

    # بديل: Inkscape
    import subprocess, tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([
            "inkscape", svg_path,
            f"--export-width={CARD_W}",
            f"--export-height={CARD_H}",
            f"--export-filename={tmp_path}"
        ], check=True, capture_output=True)
        img = Image.open(tmp_path).convert("RGBA")
        os.unlink(tmp_path)
        return img
    except Exception:
        pass

    raise RuntimeError("تعذّر تحويل SVG — تأكد من تثبيت cairosvg أو inkscape")


# ── توليد QR Code ─────────────────────────────────────────────────────────────
def generate_qr(url: str, size: int) -> Image.Image:
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=1,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img.resize((size, size), Image.LANCZOS)


# ── معالجة الصورة البيومترية ──────────────────────────────────────────────────
async def process_photo_biometric(image_bytes: bytes) -> Image.Image:
    """إزالة الخلفية + قص ذكي للوجه"""
    b64 = base64.b64encode(image_bytes).decode()
    data_uri = f"data:image/jpeg;base64,{b64}"
    try:
        result = await asyncio.to_thread(
            fal_client.subscribe,
            "fal-ai/birefnet/v2",
            arguments={"image_url": data_uri, "model": "Portrait",
                       "operating_resolution": "1024x1024"},
        )
        resp = requests.get(result["image"]["url"], timeout=30)
        resp.raise_for_status()
        cutout = Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        raise HTTPException(500, f"خطأ في معالجة الصورة: {str(e)}")

    # خلفية بيضاء
    bg = Image.new("RGBA", cutout.size, (255, 255, 255, 255))
    final = Image.alpha_composite(bg, cutout).convert("RGB")

    # قص ذكي للوجه
    img_arr = np.array(final)
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

    ih, iw = img_arr.shape[:2]
    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        fx, fy, fw, fh = faces[0]
        cx = fx + fw // 2
        crop_h = int(fh / 0.75)
        crop_w = crop_h  # مربع
        headroom = int(crop_h * 0.08)
        top  = max(0, fy - headroom)
        left = max(0, cx - crop_w // 2)
        top  = min(top,  ih - crop_h)
        left = min(left, iw - crop_w)
        final = final.crop((left, top, left+crop_w, top+crop_h))
    else:
        # قص مربع من المنتصف
        side = min(iw, ih)
        final = final.crop(((iw-side)//2, 0, (iw+side)//2, side))

    return final.resize((PHOTO_W, PHOTO_H), Image.LANCZOS)


def paste_rounded(base: Image.Image, overlay: Image.Image,
                  x: int, y: int, radius: int) -> Image.Image:
    """لصق صورة بزوايا مدورة"""
    mask = Image.new("L", overlay.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, overlay.width-1, overlay.height-1],
                        radius=radius, fill=255)
    result = base.copy()
    result.paste(overlay.convert("RGBA"), (x, y), mask)
    return result


# ── نقطة النهاية الرئيسية ────────────────────────────────────────────────────
async def generate_family_card(
    photo:              UploadFile,
    husband_name_ar:    str,
    husband_name_fr:    str,
    husband_cnie:       str,
    husband_birth_date: str,
    husband_birth_place:str,
    husband_reg_num:    str,
    wife_name_ar:       str,
    wife_name_fr:       str,
    wife_cnie:          str,
    wife_birth_date:    str,
    wife_birth_place:   str,
    wife_reg_num:       str,
    phone:              str,
    address_ar:         str,
    address_fr:         str,
    reg_num_1:          str,
    reg_num_2:          str,
    card_ref:           str,
    google_drive_url:   str,
    svg_template_path:  str = "family_card_template.svg",
) -> bytes:

    # 1. تحويل SVG إلى خلفية
    background = svg_to_background(svg_template_path)
    card = background.convert("RGB")
    draw = ImageDraw.Draw(card)

    # 2. معالجة الصورة البيومترية
    photo_bytes = await photo.read()
    person_photo = await process_photo_biometric(photo_bytes)
    card = paste_rounded(card, person_photo, PHOTO_X, PHOTO_Y, PHOTO_RADIUS)
    draw = ImageDraw.Draw(card)

    # 3. توليد QR Code
    if google_drive_url and google_drive_url.strip():
        qr_img = generate_qr(google_drive_url.strip(), QR_W)
        card.paste(qr_img, (QR_X, QR_Y))
        draw = ImageDraw.Draw(card)

    # 4. كتابة حقول النص
    data = {
        "husband_name_ar":     husband_name_ar,
        "husband_name_fr":     husband_name_fr,
        "husband_cnie":        husband_cnie,
        "husband_birth_date":  husband_birth_date,
        "husband_birth_place": husband_birth_place,
        "husband_reg_num":     husband_reg_num,
        "wife_name_ar":        wife_name_ar,
        "wife_name_fr":        wife_name_fr,
        "wife_cnie":           wife_cnie,
        "wife_birth_date":     wife_birth_date,
        "wife_birth_place":    wife_birth_place,
        "wife_reg_num":        wife_reg_num,
        "phone":               phone,
        "address_ar":          address_ar,
        "address_fr":          address_fr,
        "reg_num_1":           reg_num_1,
        "reg_num_2":           reg_num_2,
        "card_ref":            card_ref,
    }
    for field_name, (x, y, align, size, weight) in TEXT_FIELDS.items():
        value = data.get(field_name, "")
        draw_text_field(draw, value, x, y, align, size, weight)

    # 5. تصدير JPG عالي الجودة
    buf = io.BytesIO()
    card.save(buf, format="JPEG", quality=97,
              dpi=(300, 300), optimize=True)
    return buf.getvalue()
