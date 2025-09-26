import os
from pathlib import Path
from PIL import Image
import torch
import fitz  
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from paddleocr import PPStructureV3

INPUT_FILE  = r""   
OUT_DIR     = r""
DET_MODEL   = "microsoft/table-transformer-detection"
DET_THRESH  = 0.7
EXPAND_CM   = 0.30
DEFAULT_DPI = 300.0
PDF_RENDER_DPI = 300  
USE_GPU     = torch.cuda.is_available()

os.makedirs(OUT_DIR, exist_ok=True)


def crop_pil(img: Image.Image, box):
    """box = (y1, x1, y2, x2)"""
    y1, x1, y2, x2 = box
    return img.crop((x1, y1, x2, y2))


def infer_image_dpi(pil_img: Image.Image, default: float = DEFAULT_DPI) -> float:
    dpi = pil_img.info.get("dpi")
    if isinstance(dpi, tuple) and len(dpi) >= 1:
        try:
            xdpi = float(dpi[0])
            if xdpi > 0:
                return xdpi
        except Exception:
            pass
    return float(default)


def cm_to_px(cm: float, dpi: float) -> int:
    return max(1, int(round(cm * dpi / 2.54)))


def expand_box_cm_on_page(box, page_w, page_h, cm_each_side, dpi):
    y1, x1, y2, x2 = map(int, box)
    pad = cm_to_px(cm_each_side, dpi)
    y1e = max(0, y1 - pad)
    x1e = max(0, x1 - pad)
    y2e = min(page_h, y2 + pad)
    x2e = min(page_w, x2 + pad)
    if y2e - y1e < 2:
        y2e = min(page_h, y1e + 2)
    if x2e - x1e < 2:
        x2e = min(page_w, x1e + 2)
    return (y1e, x1e, y2e, x2e)


def render_pdf_to_images(pdf_path, dpi=300):
    """Yield (page_index, PIL.Image) for each page."""
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for pno in range(len(doc)):
        page = doc[pno]
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        yield pno, img
    doc.close()


# ---- Load detection model ----
device = torch.device("cuda" if USE_GPU else "cpu")
det_processor = AutoImageProcessor.from_pretrained(DET_MODEL)
det_model = TableTransformerForObjectDetection.from_pretrained(DET_MODEL).to(device).eval()

# ---- Load PaddleOCR structure model ----
pipeline = PPStructureV3(device="gpu" if USE_GPU else "cpu")

# ---- Handle input ----
suffix = Path(INPUT_FILE).suffix.lower()
if suffix == ".pdf":
    pages = render_pdf_to_images(INPUT_FILE, dpi=PDF_RENDER_DPI)
else:
    img = Image.open(INPUT_FILE).convert("RGB")
    pages = [(0, img)]


# ---- Detect tables per page ----
for page_idx, page_img in pages:
    W, H = page_img.size
    dpi = infer_image_dpi(page_img, default=DEFAULT_DPI)

    inputs = det_processor(images=page_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = det_model(**inputs)

    detections = det_processor.post_process_object_detection(
        outputs, threshold=DET_THRESH, target_sizes=[(H, W)]
    )[0]

    boxes  = detections["boxes"].cpu().numpy().tolist()
    scores = detections["scores"].cpu().numpy().tolist()

    if not boxes:
        print(f"Page {page_idx}: No tables detected.")
        continue

    for i, (b, sc) in enumerate(zip(boxes, scores)):
        x1, y1, x2, y2 = map(int, b)
        box_yx = (y1, x1, y2, x2)

        y1e, x1e, y2e, x2e = expand_box_cm_on_page(box_yx, W, H, EXPAND_CM, dpi)

        crop_img = crop_pil(page_img, (y1e, x1e, y2e, x2e))
        crop_path = os.path.join(OUT_DIR, f"page_{page_idx:03d}_table_{i:02d}.png")
        crop_img.save(crop_path)
        print(f"Page {page_idx} - Table {i} cropped: {crop_path} (score={sc:.3f})")

        # ---- Parse with PaddleOCR ----
        output = pipeline.predict(crop_path)
        for j, res in enumerate(output):
            res.print()
            res.save_to_xlsx(save_path=OUT_DIR)

        print(f"Parsed results saved in {OUT_DIR}")
