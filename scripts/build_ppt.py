# -*- coding: utf-8 -*-
"""Build the weekly progress PPTX using python-pptx (node/pptxgenjs unavailable on this machine)."""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = r"C:\Users\User1\AppData\Local\Temp\claude\C--Users-User1-pneumo-dataset\cc54ea32-dde3-492d-9e43-7954effca9f6\scratchpad\ppt_assets"

# Midnight Executive palette
NAVY = RGBColor(0x1E, 0x27, 0x61)
ICE = RGBColor(0xCA, 0xDC, 0xFC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
DARK_TEXT = RGBColor(0x22, 0x24, 0x33)
MUTED = RGBColor(0x6B, 0x72, 0x8A)
CARD_BG = RGBColor(0xF3, 0xF6, 0xFC)
ACCENT_RED = RGBColor(0xC1, 0x50, 0x2E)
GOOD_GREEN = RGBColor(0x2C, 0x5F, 0x2D)

IMG_ROI = os.path.join(ROOT, "scripts", "roi_investigation_png", "FIXED__20251028273-1.png")
IMG_BBOX = os.path.join(ROOT, "scripts", "bbox_consistency.png")
IMG_GRADCAM = os.path.join(ROOT, "gradcam_outputs", "TP__20251028201-1.png")
IMG_CURVE = os.path.join(ASSETS, "training_curve.png")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]
SW, SH = prs.slide_width, prs.slide_height


def add_slide():
    return prs.slides.add_slide(BLANK)


def set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def no_line(shape):
    shape.line.fill.background()


def add_rect(slide, x, y, w, h, fill=None, line=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    return shp


def add_text(slide, x, y, w, h, text, size=14, color=DARK_TEXT, bold=False, italic=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, font="Calibri", line_spacing=1.0,
             wrap=True):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = line
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = font
    return tb


def add_bullets(slide, x, y, w, h, items, size=14, color=DARK_TEXT, font="Calibri",
                 space_after=8, bold_lead=None, marker="•  "):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = 1.08
        if isinstance(item, tuple):
            lead, rest = item
            r1 = p.add_run()
            r1.text = marker + lead
            r1.font.bold = True
            r1.font.size = Pt(size)
            r1.font.color.rgb = color
            r1.font.name = font
            if rest:
                r2 = p.add_run()
                r2.text = rest
                r2.font.size = Pt(size)
                r2.font.color.rgb = color
                r2.font.name = font
        else:
            r = p.add_run()
            r.text = marker + item
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.name = font
    return tb


def add_pic_fit(slide, path, x, y, max_w, max_h, align_center=True):
    from PIL import Image
    with Image.open(path) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w = int(iw * ratio)
    h = int(ih * ratio)
    px = x + (max_w - w) // 2 if align_center else x
    py = y + (max_h - h) // 2 if align_center else y
    return slide.shapes.add_picture(path, px, py, width=w, height=h)


def footer(slide, page_no, dark=False):
    color = RGBColor(0xB8, 0xC2, 0xE0) if dark else MUTED
    add_text(slide, Inches(0.5), Inches(7.15), Inches(6), Inches(0.3),
              "Pneumoperitoneum CT Classification", size=9, color=color, font="Calibri")
    add_text(slide, SW - Inches(1.2), Inches(7.15), Inches(0.7), Inches(0.3),
              str(page_no), size=9, color=color, align=PP_ALIGN.RIGHT, font="Calibri")


def icon_circle(slide, cx, cy, d, fill, text_char, text_color=WHITE, size=20):
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, cx, cy, d, d)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    no_line(shp)
    shp.shadow.inherit = False
    tf = shp.text_frame
    tf.margin_left = 0; tf.margin_right = 0; tf.margin_top = 0; tf.margin_bottom = 0
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text_char
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = text_color
    r.font.name = "Calibri"
    return shp


# ============================================================ SLIDE 1: TITLE
s = add_slide()
set_bg(s, NAVY)
add_text(s, Inches(1), Inches(2.5), Inches(11.3), Inches(0.5),
          "WEEKLY PROGRESS UPDATE", size=16, color=ICE, bold=True, font="Calibri")
add_text(s, Inches(1), Inches(3.0), Inches(11.3), Inches(1.5),
          "Pneumoperitoneum CT Classification", size=40, color=WHITE, bold=True, font="Cambria")
add_text(s, Inches(1), Inches(4.15), Inches(11.3), Inches(0.6),
          "VISTA3D Encoder Transfer Learning", size=20, color=ICE, font="Calibri")
add_text(s, Inches(1), Inches(6.5), Inches(6), Inches(0.4),
          "September 3, 2026", size=14, color=RGBColor(0xB8, 0xC2, 0xE0), font="Calibri")
# simple motif: three thin circles bottom-right, echoing "encoder -> head -> logit" pipeline
for i, (dx, sz) in enumerate([(0, 60), (70, 44), (125, 30)]):
    c = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.6) + Emu(dx * 9144), Inches(1.0), Emu(sz * 9144), Emu(sz * 9144))
    c.fill.background()
    c.line.color.rgb = ICE
    c.line.width = Pt(1.5)
    c.shadow.inherit = False

# ============================================================ SLIDE 2: APPROACH
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.6), Inches(0.5), Inches(11), Inches(0.7), "Approach", size=32, color=NAVY, bold=True, font="Cambria")
add_text(s, Inches(0.6), Inches(1.25), Inches(8.5), Inches(0.4),
          "Mentor-suggested transfer learning, not training a 3D CNN from scratch", size=15, color=MUTED)

# left column: rationale bullets
add_text(s, Inches(0.6), Inches(2.0), Inches(6.2), Inches(0.4), "Why transfer learning", size=16, color=NAVY, bold=True)
add_bullets(s, Inches(0.6), Inches(2.55), Inches(6.3), Inches(3.8), [
    ("Small dataset: ", "363 CT volumes total \u2014 not enough to train a 3D CNN from scratch without severe overfitting risk"),
    ("Pretrained anatomical features: ", "VISTA3D's encoder was trained on large-scale multi-organ CT segmentation \u2014 it already knows what a liver, bowel, kidney look like"),
    ("Faster, cheaper signal: ", "a linear probe (frozen encoder + small head) gives an honest first read in hours, not days"),
    ("Selective fine-tuning: ", "only the deepest encoder stage is unfrozen later \u2014 adapts to the task without destroying pretrained features"),
], size=14, space_after=14)

# right column: simple pipeline diagram (3 boxes + arrows)
diagram_y = Inches(2.7)
box_w, box_h = Inches(1.7), Inches(1.0)
labels = ["CT\nVolume", "VISTA3D\nEncoder\n(pretrained)", "Classification\nHead", "Pneumo-\nperitoneum\nprobability"]
colors = [CARD_BG, ICE, CARD_BG, CARD_BG]
xs = [Inches(7.3), Inches(9.15), Inches(11.0), Inches(11.0)]
box1 = add_rect(s, xs[0], diagram_y, box_w, box_h, fill=CARD_BG)
add_text(s, xs[0], diagram_y, box_w, box_h, labels[0], size=12, color=DARK_TEXT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box2 = add_rect(s, xs[1], diagram_y, box_w, box_h, fill=NAVY)
add_text(s, xs[1], diagram_y, box_w, box_h, labels[1], size=12, color=WHITE, bold=True, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box3 = add_rect(s, xs[0], diagram_y + Inches(1.4), box_w, box_h, fill=CARD_BG)
add_text(s, xs[0], diagram_y + Inches(1.4), box_w, box_h, labels[2], size=12, color=DARK_TEXT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
box4 = add_rect(s, xs[1], diagram_y + Inches(1.4), box_w, box_h, fill=CARD_BG)
add_text(s, xs[1], diagram_y + Inches(1.4), box_w, box_h, labels[3], size=11, color=DARK_TEXT, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
# connectors
for (x1, y1, x2, y2) in [(xs[0]+box_w, diagram_y+box_h//2, xs[1], diagram_y+box_h//2),
                          (xs[1]+box_w//2, diagram_y+box_h, xs[1]+box_w//2, diagram_y+Inches(1.4)),
                          (xs[0]+box_w, diagram_y+Inches(1.4)+box_h//2, xs[1], diagram_y+Inches(1.4)+box_h//2)]:
    ln = s.shapes.add_connector(2, x1, y1, x2, y2)
    ln.line.color.rgb = MUTED
    ln.line.width = Pt(1.5)
add_text(s, Inches(7.3), diagram_y + Inches(1.15), Inches(3.4), Inches(0.25), "frozen \u2192 later partially unfrozen",
          size=10, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
add_text(s, Inches(7.3), diagram_y + Inches(2.6), Inches(3.4), Inches(0.3), "only this block is trained",
          size=10, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
footer(s, 2)

# ============================================================ SLIDE 3: PREPROCESSING
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.6), Inches(0.4), Inches(11), Inches(0.6), "Preprocessing Pipeline", size=30, color=NAVY, bold=True, font="Cambria")
add_text(s, Inches(0.6), Inches(1.05), Inches(11), Inches(0.35),
          "Every volume goes through the same 5-step chain before reaching the model", size=14, color=MUTED)

# 5-step horizontal flow
steps = [
    ("1", "Reorient", "Canonicalize to RAS\n(5 orientation codes seen)"),
    ("2", "Resample", "1.5mm isotropic\n(VISTA3D's own convention)"),
    ("3", "ROI crop", "VISTA3D-segmented,\nfixed 280mm window,\ndiaphragm-biased"),
    ("4", "Intensity scale", "HU → [0,1] using\nVISTA3D's exact window"),
    ("5", "Resize", "128³ / 224³\nfinal tensor"),
]
step_w = Inches(2.15)
gap = Inches(0.15)
start_x = Inches(0.6)
step_y = Inches(1.65)
for i, (num, title, desc) in enumerate(steps):
    x = start_x + i * (step_w + gap)
    card = add_rect(s, x, step_y, step_w, Inches(1.9), fill=CARD_BG)
    icon_circle(s, x + Inches(0.15), step_y + Inches(0.15), Inches(0.45), NAVY, num, size=16)
    add_text(s, x + Inches(0.15), step_y + Inches(0.72), step_w - Inches(0.3), Inches(0.35), title,
              size=14, color=NAVY, bold=True)
    add_text(s, x + Inches(0.15), step_y + Inches(1.1), step_w - Inches(0.3), Inches(0.75), desc,
              size=10.5, color=DARK_TEXT, line_spacing=1.1)
    if i < len(steps) - 1:
        arrow_x = x + step_w + Emu(int(gap / 2) - 45000)
        arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, arrow_x, step_y + Inches(0.85), Emu(140000), Inches(0.2))
        arr.fill.solid(); arr.fill.fore_color.rgb = MUTED; no_line(arr); arr.shadow.inherit = False

# QC visualization: real artifact from this week's work
add_text(s, Inches(0.6), Inches(3.85), Inches(6), Inches(0.35),
          "QC spot-check (real output, not a mockup)", size=13, color=NAVY, bold=True)
add_pic_fit(s, IMG_ROI, Inches(0.6), Inches(4.2), Inches(12.1), Inches(2.9), align_center=True)
add_text(s, Inches(0.6), Inches(7.05), Inches(11), Inches(0.3),
          "Top row: original CT + ROI bounding box overlay. Bottom row: final preprocessed 224³ tensor the model actually sees.",
          size=9.5, color=MUTED, italic=True)
footer(s, 3)

# ============================================================ SLIDE 4: CHALLENGES - DATA/PIPELINE
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.6), Inches(0.4), Inches(11.5), Inches(0.6), "Challenges: Data & Pipeline Bugs", size=30, color=NAVY, bold=True, font="Cambria")
add_text(s, Inches(0.6), Inches(1.05), Inches(11.5), Inches(0.4),
          "Two real bugs found this week — both caught by looking at images, not just summary statistics", size=14, color=MUTED, italic=True)

card_y = Inches(1.65)
card_h = Inches(2.55)
card_w = Inches(5.85)
c1 = add_rect(s, Inches(0.6), card_y, card_w, card_h, fill=CARD_BG)
icon_circle(s, Inches(0.85), card_y + Inches(0.25), Inches(0.4), ACCENT_RED, "1", size=15)
add_text(s, Inches(1.4), card_y + Inches(0.22), card_w - Inches(1.0), Inches(0.45),
          "Axis-mislabeling bug", size=15, color=NAVY, bold=True)
add_bullets(s, Inches(0.85), card_y + Inches(0.85), card_w - Inches(0.5), Inches(1.6), [
    "Audit script assumed array axis 2 = slice axis — false for 2 of 5 orientation codes seen in this dataset",
    "Silently corrupted scan-protocol classification for 77/363 volumes (21%)",
    "75 volumes wrongly labeled “abdomen-only” were actually full torso scans — never got ROI-cropped",
], size=12.5, space_after=8)

c2 = add_rect(s, Inches(6.75), card_y, card_w, card_h, fill=CARD_BG)
icon_circle(s, Inches(7.0), card_y + Inches(0.25), Inches(0.4), ACCENT_RED, "2", size=15)
add_text(s, Inches(7.55), card_y + Inches(0.22), card_w - Inches(1.0), Inches(0.45),
          "ROI-crop shortcut-learning gap", size=15, color=NAVY, bold=True)
add_bullets(s, Inches(7.0), card_y + Inches(0.85), card_w - Inches(0.5), Inches(1.6), [
    "Original crop: cropped z-extent (502mm) vs. naturally-small scans (250mm) — completely non-overlapping",
    "Model could trivially learn “scan protocol” instead of pathology",
    "Fixed with a centroid-anchored, diaphragm-biased 280mm fixed window — verified the gap closed",
], size=12.5, space_after=8)

add_text(s, Inches(0.6), Inches(4.35), Inches(6), Inches(0.3), "Before / after fix: z-extent distributions", size=12.5, color=NAVY, bold=True)
add_pic_fit(s, IMG_BBOX, Inches(0.6), Inches(4.68), Inches(6.2), Inches(2.55), align_center=True)
add_bullets(s, Inches(7.1), Inches(4.68), Inches(5.6), Inches(2.4), [
    ("Both bugs caught by ", "visually spot-checking preprocessed volumes and bounding boxes"),
    ("Neither showed up as an obvious statistical outlier ", "in aggregate summary tables first"),
    ("Take-away: ", "visual QC on a data pipeline is not optional for medical imaging work"),
], size=13, space_after=12)
footer(s, 4)

# ============================================================ SLIDE 5: CHALLENGES - INFRASTRUCTURE
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.6), Inches(0.4), Inches(11.5), Inches(0.6), "Challenges: Infrastructure & Training", size=30, color=NAVY, bold=True, font="Cambria")

rows = [
    ("MONAI Label server incompatibility", "Version-mismatch bugs in monailabel's BundleInferTask — pivoted to loading the VISTA3D bundle (model.pt + configs) directly instead"),
    ("GPU migration: RTX 4090 → RTX 3060 (12GB)", "Original plan assumed 24GB; 12GB forced batch_size=1 + gradient accumulation (effective batch=4), verified empirically with memory probes"),
    ("Early fine-tuning instability", "Unfreezing stage 4 without warmup caused wild sens/spec swings epoch-to-epoch (0.86/0.15 → 0.43/0.67 → 0.10/1.0) — fixed with 2-epoch LR warmup + cosine decay"),
    ("Validation-set overfitting from repeated tuning", "Checkpoint selection + threshold tuning both reused the same 54-volume val set — held-out test set revealed a real, honest performance gap"),
]
row_y = Inches(1.3)
row_h = Inches(1.42)
for i, (title, desc) in enumerate(rows):
    y = row_y + i * row_h
    icon_circle(s, Inches(0.6), y + Inches(0.12), Inches(0.55), NAVY, str(i + 1), size=18)
    add_text(s, Inches(1.4), y, Inches(11.0), Inches(0.4), title, size=15, color=NAVY, bold=True)
    add_text(s, Inches(1.4), y + Inches(0.42), Inches(11.2), Inches(0.85), desc, size=13, color=DARK_TEXT, line_spacing=1.15)
footer(s, 5)

# ============================================================ SLIDE 6: RESULTS
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.5), Inches(0.3), Inches(11.5), Inches(0.55), "Results So Far", size=28, color=NAVY, bold=True, font="Cambria")

# results table
tbl_x, tbl_y, tbl_w, tbl_h = Inches(0.5), Inches(0.9), Inches(8.6), Inches(1.85)
gframe = s.shapes.add_table(4, 6, tbl_x, tbl_y, tbl_w, tbl_h)
tbl = gframe.table
headers = ["Model", "AUROC", "AUPRC", "Sens.", "Spec.", "Acc."]
data_rows = [
    ("Linear probe (frozen)", "0.740", "0.747", "0.619", "0.909", "0.796"),
    ("Fine-tuned (stage 4)", "0.775", "0.795", "0.667", "0.879", "0.796"),
    ("HELD-OUT TEST", "0.650", "0.582", "0.565", "0.719", "0.655"),
]
col_widths = [Inches(2.6), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)]
for i, cw in enumerate(col_widths):
    tbl.columns[i].width = cw
for j, h in enumerate(headers):
    cell = tbl.cell(0, j)
    cell.text = h
    cell.fill.solid(); cell.fill.fore_color.rgb = NAVY
    p = cell.text_frame.paragraphs[0]; p.alignment = PP_ALIGN.CENTER
    r = p.runs[0]; r.font.color.rgb = WHITE; r.font.bold = True; r.font.size = Pt(12); r.font.name = "Calibri"
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
for i, row in enumerate(data_rows):
    is_test = (i == 2)
    for j, val in enumerate(row):
        cell = tbl.cell(i + 1, j)
        cell.text = val
        cell.fill.solid()
        cell.fill.fore_color.rgb = RGBColor(0xFD, 0xEE, 0xE9) if is_test else (CARD_BG if i % 2 == 0 else WHITE)
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER
        r = p.runs[0]
        r.font.size = Pt(12)
        r.font.bold = is_test
        r.font.color.rgb = ACCENT_RED if is_test else DARK_TEXT
        r.font.name = "Calibri"
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
# remove default table style borders look via keeping simple; python-pptx default banding is fine.

add_text(s, Inches(0.5), Inches(2.9), Inches(8.6), Inches(0.6),
          "Val AUPRC dropped ~0.21 on the untouched test set — the val set was reused for both checkpoint "
          "selection and threshold tuning, and both splits are tiny (54/55 volumes). Test is the trustworthy number.",
          size=11.5, color=MUTED, italic=True, line_spacing=1.15)

add_text(s, Inches(9.3), Inches(0.9), Inches(3.5), Inches(0.35), "Also tried", size=13, color=NAVY, bold=True)
add_bullets(s, Inches(9.3), Inches(1.25), Inches(3.5), Inches(1.3), [
    ("Ensemble ", "(avg of linear probe + fine-tune): val AUPRC 0.764 — did not beat fine-tune alone, not used"),
], size=11.5, space_after=6)

# training curve + gradcam side by side
add_text(s, Inches(0.5), Inches(3.65), Inches(6), Inches(0.3), "Training trajectory (real logged data)", size=12.5, color=NAVY, bold=True)
add_pic_fit(s, IMG_CURVE, Inches(0.5), Inches(3.98), Inches(6.3), Inches(3.0), align_center=True)

add_text(s, Inches(7.0), Inches(3.65), Inches(5.8), Inches(0.3), "Grad-CAM — true positive (prob=0.87)", size=12.5, color=NAVY, bold=True)
add_pic_fit(s, IMG_GRADCAM, Inches(7.0), Inches(3.98), Inches(5.8), Inches(2.55), align_center=True)
add_text(s, Inches(7.0), Inches(6.58), Inches(5.8), Inches(0.55),
          "Hotspots concentrate near colon & liver when landing on a segmented organ — anatomically relevant "
          "to pneumoperitoneum (bowel source, subphrenic space).", size=10, color=MUTED, italic=True, line_spacing=1.1)
footer(s, 6)

# ============================================================ SLIDE 7: NEXT STEPS
s = add_slide()
set_bg(s, WHITE)
add_text(s, Inches(0.6), Inches(0.5), Inches(11.5), Inches(0.6), "Next Steps", size=32, color=NAVY, bold=True, font="Cambria")
add_text(s, Inches(0.6), Inches(1.2), Inches(11), Inches(0.4),
          "Getting a more trustworthy performance estimate and a real architecture comparison", size=14, color=MUTED)

steps7 = [
    ("3-Fold Patient-Level Cross-Validation",
     "Patient-grouped StratifiedGroupKFold (multi-scan patients kept intact within one fold) — "
     "linear probe + fine-tune per fold, both with early stopping. Orchestrator built and wiring-verified; ready to launch overnight."),
    ("3D ResNet Baseline Comparison",
     "Same preprocessing pipeline, different pretrained backbone (medical-pretrained 3D ResNet) — "
     "tests whether VISTA3D's encoder is actually the right choice, or any reasonable pretrained 3D CNN gets similar results."),
    ("Bootstrap Confidence Intervals",
     "Resample the held-out test set to put an honest interval around AUROC/AUPRC — "
     "a single-point estimate on 55 volumes needs an uncertainty range to be meaningfully reported."),
]
y0 = Inches(2.0)
row_h = Inches(1.65)
for i, (title, desc) in enumerate(steps7):
    y = y0 + i * row_h
    card = add_rect(s, Inches(0.6), y, Inches(11.9), Inches(1.4), fill=CARD_BG)
    icon_circle(s, Inches(0.85), y + Inches(0.42), Inches(0.55), NAVY, str(i + 1), size=18)
    add_text(s, Inches(1.7), y + Inches(0.18), Inches(10.6), Inches(0.4), title, size=16, color=NAVY, bold=True)
    add_text(s, Inches(1.7), y + Inches(0.62), Inches(10.6), Inches(0.7), desc, size=12.5, color=DARK_TEXT, line_spacing=1.15)
footer(s, 7)

# ============================================================ SLIDE 8: SUMMARY
s = add_slide()
set_bg(s, NAVY)
add_text(s, Inches(0.8), Inches(0.6), Inches(11), Inches(0.6), "Summary & Takeaways", size=32, color=WHITE, bold=True, font="Cambria")

takeaways = [
    ("Real signal, not yet strong",
     "Held-out test AUROC ≈ 0.65 — clearly above chance (0.50), but this is an early-stage classifier, not a validated tool."),
    ("Rigorous methodology this week",
     "Two real bugs caught via visual QC (not just summary stats); test set touched exactly once, honestly, after all tuning was finished."),
    ("Clear path forward",
     "3-fold cross-validation, a second pretrained backbone for comparison, and confidence intervals — all scoped and ready to run."),
]
card_y = Inches(1.75)
card_h = Inches(1.55)
gap = Inches(0.25)
for i, (title, desc) in enumerate(takeaways):
    y = card_y + i * (card_h + gap)
    card = add_rect(s, Inches(0.8), y, Inches(11.7), card_h, fill=RGBColor(0x28, 0x32, 0x78))
    icon_circle(s, Inches(1.1), y + Inches(0.42), Inches(0.7), ICE, str(i + 1), text_color=NAVY, size=22)
    add_text(s, Inches(2.15), y + Inches(0.2), Inches(10.1), Inches(0.45), title, size=18, color=WHITE, bold=True)
    add_text(s, Inches(2.15), y + Inches(0.72), Inches(10.1), Inches(0.7), desc, size=13, color=ICE, line_spacing=1.2)
footer(s, 8, dark=True)

out_path = os.path.join(ROOT, "Pneumoperitoneum_CT_Weekly_Update.pptx")
tmp_path = os.path.join(ROOT, "_tmp_build.pptx")
prs.save(tmp_path)
try:
    os.replace(tmp_path, out_path)
    print(f"Saved {out_path}")
except PermissionError:
    print(f"Target locked (likely open in PowerPoint) -- saved to {tmp_path} instead")

