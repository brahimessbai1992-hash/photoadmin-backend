"""
cnss_card_api.py
================
يولّد بطاقة AMO TADAMON للصندوق الوطني للضمان الاجتماعي (CNSS)
بجودة طباعة عالية عبر رسم النصوص فوق الخلفية الأصلية.

أبعاد الخلفية المرجعية: 2000 × 1294 px
"""

import io
import os
from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont
from fastapi import UploadFile


# ── الألوان ───────────────────────────────────────────────────
DARK_BLUE  = (26,  60, 120)   # النصوص الثابتة (عربي + فرنسي)
TEAL       = (32, 178, 170)   # القيم المتغيرة (أرقام + أسماء)
TEAL_BIG   = (32, 178, 170)   # رقم التسجيل الكبير

# ── مسارات الخطوط (على Railway) ──────────────────────────────
# يمكن تغييرها حسب المسار الفعلي على الخادم
_FONT_DIR = Path(os.getenv("FONT_DIR", "/usr/share/fonts/truetype"))

def _find_font(names: list[str], fallback: str = "DejaVuSans.ttf") -> str:
    """يبحث عن أول خط متاح من القائمة"""
    for name in names:
        for f in _FONT_DIR.rglob(name):
            return str(f)
    # احتياط: DejaVu
    for f in _FONT_DIR.rglob(fallback):
        return str(f)
    return None

FONT_AR_REGULAR = _find_font(["Amiri-Regular.ttf", "NotoNaskhArabic-Regular.ttf", "Arial.ttf"])
FONT_AR_BOLD    = _find_font(["Amiri-Bold.ttf",    "NotoNaskhArabic-Bold.ttf",    "Arial Bold.ttf"])
FONT_FR_REGULAR = _find_font(["DejaVuSans.ttf",    "Arial.ttf"])
FONT_FR_BOLD    = _find_font(["DejaVuSans-Bold.ttf","Arial Bold.ttf"])

# ── مقياس البطاقة ─────────────────────────────────────────────
# الصورة المرجعية 2000×1294 px — كل الإحداثيات بهذا المقياس
W, H = 2000, 1294


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def _ar(text: str) -> str:
    """تشكيل النص العربي للعرض الصحيح"""
    try:
        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:
        return text


def _draw_text_right(draw, text, x_right, y, font, fill):
    """رسم نص محاذاة يمين — x_right هو الحد الأيمن"""
    bbox = draw.textbbox((0, 0), text, font=font)
    w    = bbox[2] - bbox[0]
    draw.text((x_right - w, y), text, font=font, fill=fill)


def _draw_text_center(draw, text, x_center, y, font, fill):
    """رسم نص في المنتصف"""
    bbox = draw.textbbox((0, 0), text, font=font)
    w    = bbox[2] - bbox[0]
    draw.text((x_center - w // 2, y), text, font=font, fill=fill)


def generate_cnss_card(
    # البيانات الأساسية
    reg_num:        str,   # رقم التسجيل  906280021
    nom_ar:         str,   # الاسم العائلي بالعربية
    prenom_ar:      str,   # الاسم الشخصي بالعربية
    birth_date:     str,   # تاريخ الازدياد  01-01-1970
    cin:            str,   # رقم البطاقة الوطنية
    reg_date:       str,   # تاريخ التسجيل  08-02-2026
    nom_fr:         str,   # Nom en français
    prenom_fr:      str,   # Prénom en français
    bg_path:        str,   # مسار صورة الخلفية PNG
    output_scale:   float = 1.0,   # 1.0 = 2000px عرض
) -> bytes:
    """
    يولّد بطاقة CNSS كاملة ويُرجعها كـ bytes (JPEG جودة 97).
    """
    # ── تحميل الخلفية ─────────────────────────────────────────
    bg = Image.open(bg_path).convert("RGB")
    if output_scale != 1.0:
        nw = int(bg.width  * output_scale)
        nh = int(bg.height * output_scale)
        bg = bg.resize((nw, nh), Image.LANCZOS)

    s = bg.width / W   # معامل التحجيم

    draw = ImageDraw.Draw(bg)

    # ── تعريف الخطوط ──────────────────────────────────────────
    # العناوين الثابتة (شهادة التسجيل...)
    f_title_ar = _font(FONT_AR_BOLD,    int(48 * s))
    # AMO TADAMON
    f_amo      = _font(FONT_FR_BOLD,    int(46 * s))
    # رقم التسجيل الكبير
    f_reg_big  = _font(FONT_FR_BOLD,    int(90 * s))
    # تسميات الحقول (Nom:, رقم التسجيل...)
    f_label_fr = _font(FONT_FR_BOLD,    int(38 * s))
    f_label_ar = _font(FONT_AR_BOLD,    int(38 * s))
    # قيم الحقول
    f_val_fr   = _font(FONT_FR_BOLD,    int(44 * s))
    f_val_ar   = _font(FONT_AR_BOLD,    int(44 * s))
    # التواريخ والأرقام
    f_val_num  = _font(FONT_FR_BOLD,    int(42 * s))

    # ═══════════════════════════════════════════════════════════
    #  السطر 1: شهادة التسجيل... (العنوان الكبير)
    # ═══════════════════════════════════════════════════════════
    title1_ar = _ar("شهادة التسجيل بنظام التأمين الاجباري الاساسي عن المرض الخاص")
    title2_ar = _ar("بالاشخاص غير القادرين على تحمل واجبات الاشتراك")

    _draw_text_right(draw, title1_ar, int(1960 * s), int(42  * s), f_title_ar, DARK_BLUE)
    _draw_text_right(draw, title2_ar, int(1960 * s), int(100 * s), f_title_ar, DARK_BLUE)

    # ═══════════════════════════════════════════════════════════
    #  AMO TADAMON
    # ═══════════════════════════════════════════════════════════
    _draw_text_center(draw, "AMO TADAMON", int(1100 * s), int(188 * s), f_amo, DARK_BLUE)

    # ═══════════════════════════════════════════════════════════
    #  رقم التسجيل — سطر كامل
    # ═══════════════════════════════════════════════════════════
    # يمين: رقم التسجيل (عربي)
    _draw_text_right(draw, _ar("رقم التسجيل"), int(1960 * s), int(285 * s), f_label_ar, DARK_BLUE)
    # وسط: الرقم نفسه (كبير تيل)
    _draw_text_center(draw, reg_num,            int(1060 * s), int(268 * s), f_reg_big,  TEAL_BIG)
    # يسار: N° d'immatriculation
    draw.text((int(58 * s), int(295 * s)), "N° d'immatriculation", font=f_label_fr, fill=DARK_BLUE)

    # ═══════════════════════════════════════════════════════════
    #  الاسم العائلي
    # ═══════════════════════════════════════════════════════════
    y_nom = int(420 * s)
    _draw_text_right(draw, _ar("الاسم العائلي:"), int(1960 * s), y_nom, f_label_ar, DARK_BLUE)
    _draw_text_right(draw, _ar(nom_ar),           int(1440 * s), y_nom, f_val_ar,   TEAL)
    draw.text((int(58  * s), y_nom), "Nom:",      font=f_label_fr, fill=DARK_BLUE)
    draw.text((int(230 * s), y_nom), nom_fr,      font=f_val_fr,   fill=TEAL)

    # ═══════════════════════════════════════════════════════════
    #  الاسم الشخصي
    # ═══════════════════════════════════════════════════════════
    y_prenom = int(530 * s)
    _draw_text_right(draw, _ar("الاسم الشخصي:"), int(1960 * s), y_prenom, f_label_ar, DARK_BLUE)
    _draw_text_right(draw, _ar(prenom_ar),        int(1440 * s), y_prenom, f_val_ar,   TEAL)
    draw.text((int(58  * s), y_prenom), "Prénom:", font=f_label_fr, fill=DARK_BLUE)
    draw.text((int(260 * s), y_prenom), prenom_fr, font=f_val_fr,   fill=TEAL)

    # ═══════════════════════════════════════════════════════════
    #  تاريخ الازدياد
    # ═══════════════════════════════════════════════════════════
    y_birth = int(640 * s)
    _draw_text_right(draw, _ar("تاريخ الازدياد:"),    int(1960 * s), y_birth, f_label_ar, DARK_BLUE)
    _draw_text_center(draw, birth_date,                int(1060 * s), y_birth, f_val_num,  TEAL)
    draw.text((int(58 * s), y_birth), "Date de naissance:", font=f_label_fr, fill=DARK_BLUE)

    # ═══════════════════════════════════════════════════════════
    #  رقم البطاقة الوطنية CIN
    # ═══════════════════════════════════════════════════════════
    y_cin = int(750 * s)
    _draw_text_right(draw, _ar("ب.ت.و:"),    int(1960 * s), y_cin, f_label_ar, DARK_BLUE)
    _draw_text_center(draw, cin,              int(1060 * s), y_cin, f_val_num,  TEAL)
    draw.text((int(58 * s), y_cin), "C.I.N:", font=f_label_fr, fill=DARK_BLUE)

    # ═══════════════════════════════════════════════════════════
    #  تاريخ التسجيل
    # ═══════════════════════════════════════════════════════════
    y_regdate = int(860 * s)
    _draw_text_right(draw, _ar("تاريخ التسجيل:"),       int(1960 * s), y_regdate, f_label_ar, DARK_BLUE)
    _draw_text_center(draw, reg_date,                    int(1060 * s), y_regdate, f_val_num,  TEAL)
    draw.text((int(58 * s), y_regdate), "Date d'immatriculation:", font=f_label_fr, fill=DARK_BLUE)

    # ── إخراج الصورة ──────────────────────────────────────────
    buf = io.BytesIO()
    bg.save(buf, format="JPEG", quality=97, dpi=(300, 300))
    return buf.getvalue()
