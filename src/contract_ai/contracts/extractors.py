from __future__ import annotations

import uuid
from pathlib import Path

import cv2
import fitz
import numpy as np
import pdfplumber
import pytesseract
from PIL import Image

from contract_ai.common.schemas import BBox, ContractElement


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _bbox_area(bbox: BBox) -> float:
    return max(0.0, bbox.x1 - bbox.x0) * max(0.0, bbox.y1 - bbox.y0)


def bbox_overlap_ratio(a: BBox, b: BBox) -> float:
    inter_x0 = max(a.x0, b.x0)
    inter_y0 = max(a.y0, b.y0)
    inter_x1 = min(a.x1, b.x1)
    inter_y1 = min(a.y1, b.y1)
    inter_w = max(0.0, inter_x1 - inter_x0)
    inter_h = max(0.0, inter_y1 - inter_y0)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    denom = min(_bbox_area(a), _bbox_area(b))
    if denom <= 0:
        return 0.0
    return inter / denom


def _pixel_to_page_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    page_width: float,
    page_height: float,
    img_width: int,
    img_height: int,
) -> BBox:
    sx = page_width / float(img_width)
    sy = page_height / float(img_height)
    return BBox(x0=x * sx, y0=y * sy, x1=(x + w) * sx, y1=(y + h) * sy)


def _page_to_pixel_bbox(
    bbox: BBox,
    page_width: float,
    page_height: float,
    img_width: int,
    img_height: int,
) -> tuple[int, int, int, int]:
    px0 = int(max(0, min(img_width - 1, round((bbox.x0 / page_width) * img_width))))
    py0 = int(max(0, min(img_height - 1, round((bbox.y0 / page_height) * img_height))))
    px1 = int(max(0, min(img_width, round((bbox.x1 / page_width) * img_width))))
    py1 = int(max(0, min(img_height, round((bbox.y1 / page_height) * img_height))))
    return px0, py0, max(1, px1 - px0), max(1, py1 - py0)


def _normalize_color(value: int | None) -> str:
    if value is None:
        return "#000000"
    r = (value >> 16) & 255
    g = (value >> 8) & 255
    b = value & 255
    return f"#{r:02x}{g:02x}{b:02x}"


def _extract_horizontal_rules(page: fitz.Page) -> list[tuple[float, float, float]]:
    rules: list[tuple[float, float, float]] = []
    for drawing in page.get_drawings():
        for item in drawing.get("items", []):
            op = item[0]
            if op == "l":
                p1, p2 = item[1], item[2]
                if abs(float(p1.y) - float(p2.y)) <= 1.6 and abs(float(p2.x) - float(p1.x)) >= 22:
                    rules.append((min(float(p1.x), float(p2.x)), max(float(p1.x), float(p2.x)), float(p1.y)))
            elif op == "re":
                rect = item[1]
                if rect.width >= 22 and rect.height <= 2.5:
                    y = float(rect.y0 + rect.y1) / 2.0
                    rules.append((float(rect.x0), float(rect.x1), y))
    return rules


def _line_has_underline(line_bbox: tuple[float, float, float, float], rules: list[tuple[float, float, float]]) -> bool:
    x0, _, x1, y1 = line_bbox
    width = max(1.0, x1 - x0)
    for rx0, rx1, ry in rules:
        if abs(ry - y1) > 3.2:
            continue
        overlap = max(0.0, min(x1, rx1) - max(x0, rx0))
        if overlap >= width * 0.45:
            return True
    return False


def extract_text_lines_with_style(page: fitz.Page, page_num: int) -> list[ContractElement]:
    text_dict = page.get_text("dict")
    horizontal_rules = _extract_horizontal_rules(page)
    page_center = page.rect.width / 2.0

    elements: list[ContractElement] = []
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text = "".join(str(span.get("text", "")) for span in spans)
            if not text.strip():
                continue

            x0, y0, x1, y1 = line["bbox"]
            font_sizes = [float(span.get("size", 0.0)) for span in spans if span.get("size")]
            font_size = max(font_sizes) if font_sizes else max(8.0, float(y1 - y0))

            fonts = [str(span.get("font", "")) for span in spans]
            bold = any("bold" in f.lower() for f in fonts)
            italic = any("italic" in f.lower() or "oblique" in f.lower() for f in fonts)
            centered = abs(((x0 + x1) / 2.0) - page_center) <= page.rect.width * 0.08
            underline = _line_has_underline((x0, y0, x1, y1), horizontal_rules)

            span_meta: list[dict] = []
            for span in spans:
                span_font = str(span.get("font", ""))
                span_meta.append(
                    {
                        "text": str(span.get("text", "")),
                        "font": span_font,
                        "font_size": float(span.get("size", font_size)),
                        "color": _normalize_color(span.get("color")),
                        "bold": "bold" in span_font.lower(),
                        "italic": ("italic" in span_font.lower()) or ("oblique" in span_font.lower()),
                    }
                )

            alignment = "center" if centered else "left"
            elements.append(
                ContractElement(
                    element_id=_new_id("textline"),
                    element_type="text_line",
                    page_number=page_num,
                    bbox=BBox(x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1)),
                    order_index=-1,
                    text=text,
                    style={
                        "alignment": alignment,
                        "font_size": round(font_size, 2),
                        "bold": bold,
                        "italic": italic,
                        "underline": underline,
                        "source": "pdf_text",
                    },
                    metadata={"spans": span_meta},
                )
            )
    return elements


def render_page_to_image(page: fitz.Page, dpi: int = 220) -> Image.Image:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    return Image.frombytes(mode, [pix.width, pix.height], pix.samples)


def export_page_preview(page: fitz.Page, page_num: int, output_dir: Path, dpi: int = 140) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = render_page_to_image(page, dpi=dpi).convert("RGB")
    path = output_dir / f"page_{page_num}.png"
    image.save(path)
    return path


def _parse_conf(value: str | int | float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _is_bbox_in_regions(bbox: BBox, regions: list[BBox], threshold: float = 0.45) -> bool:
    return any(bbox_overlap_ratio(bbox, region) >= threshold for region in regions)


def extract_ocr_lines(
    page: fitz.Page,
    page_num: int,
    min_confidence: float = 30.0,
    excluded_regions: list[BBox] | None = None,
) -> list[ContractElement]:
    excluded_regions = excluded_regions or []
    image = render_page_to_image(page, dpi=300).convert("RGB")
    img_w, img_h = image.size
    page_w, page_h = float(page.rect.width), float(page.rect.height)

    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
    grouped: dict[tuple[int, int, int], dict] = {}
    total = len(data.get("text", []))

    for i in range(total):
        raw = str(data["text"][i] or "")
        text = raw.strip()
        conf = _parse_conf(data["conf"][i])
        if not text or conf < min_confidence:
            continue

        left, top = int(data["left"][i]), int(data["top"][i])
        width, height = int(data["width"][i]), int(data["height"][i])
        word_bbox = _pixel_to_page_bbox(left, top, width, height, page_w, page_h, img_w, img_h)
        if _is_bbox_in_regions(word_bbox, excluded_regions):
            continue

        key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))

        slot = grouped.setdefault(
            key,
            {
                "words": [],
                "x0": left,
                "y0": top,
                "x1": left + width,
                "y1": top + height,
                "conf": [],
            },
        )
        slot["words"].append((left, text))
        slot["x0"] = min(slot["x0"], left)
        slot["y0"] = min(slot["y0"], top)
        slot["x1"] = max(slot["x1"], left + width)
        slot["y1"] = max(slot["y1"], top + height)
        slot["conf"].append(conf)

    elements: list[ContractElement] = []
    for key, row in sorted(grouped.items(), key=lambda x: (x[1]["y0"], x[1]["x0"])):
        sorted_words = sorted(row["words"], key=lambda y: y[0])
        line_text = " ".join(word for _, word in sorted_words)
        if not line_text.strip():
            continue

        px0, py0, px1, py1 = row["x0"], row["y0"], row["x1"], row["y1"]
        bbox = _pixel_to_page_bbox(px0, py0, px1 - px0, py1 - py0, page_w, page_h, img_w, img_h)
        if _is_bbox_in_regions(bbox, excluded_regions, threshold=0.35):
            continue

        font_size = max(8.0, bbox.y1 - bbox.y0)
        centered = abs(((bbox.x0 + bbox.x1) / 2.0) - (page_w / 2.0)) <= page_w * 0.08

        elements.append(
            ContractElement(
                element_id=_new_id("ocrline"),
                element_type="ocr_line",
                page_number=page_num,
                bbox=bbox,
                order_index=-1,
                text=line_text,
                style={
                    "alignment": "center" if centered else "left",
                    "font_size": round(font_size, 2),
                    "bold": False,
                    "italic": False,
                    "underline": False,
                    "source": "ocr",
                },
                metadata={
                    "ocr_confidence": round(float(np.mean(row["conf"])), 2),
                    "ocr_group": {"block": key[0], "paragraph": key[1], "line": key[2]},
                },
            )
        )
    return elements


def extract_tables(pdf_path: Path, page_num: int) -> list[ContractElement]:
    elements: list[ContractElement] = []
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num - 1]
        tables = page.find_tables()
        for table in tables:
            x0, top, x1, bottom = table.bbox
            rows = table.extract() or []
            normalized_rows = [["" if c is None else str(c) for c in row] for row in rows]
            elements.append(
                ContractElement(
                    element_id=_new_id("table"),
                    element_type="table",
                    page_number=page_num,
                    bbox=BBox(x0=float(x0), y0=float(top), x1=float(x1), y1=float(bottom)),
                    order_index=-1,
                    table_data=normalized_rows,
                    style={"source": "pdf_table"},
                )
            )
    return elements


def page_has_large_embedded_image(page: fitz.Page, min_area_ratio: float = 0.85) -> bool:
    page_area = float(page.rect.width * page.rect.height)
    if page_area <= 0:
        return False
    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            if rect.get_area() / page_area >= min_area_ratio:
                return True
    return False


def extract_embedded_images(
    page: fitz.Page,
    page_num: int,
    output_dir: Path,
    max_area_ratio: float = 0.85,
) -> list[ContractElement]:
    output_dir.mkdir(parents=True, exist_ok=True)
    page_area = float(page.rect.width * page.rect.height)
    elements: list[ContractElement] = []
    image_cache: dict[int, str] = {}

    for idx, img in enumerate(page.get_images(full=True)):
        xref = img[0]
        rects = [rect for rect in page.get_image_rects(xref) if (rect.get_area() / max(page_area, 1.0)) < max_area_ratio]
        if not rects:
            continue

        if xref not in image_cache:
            try:
                base = page.parent.extract_image(xref)
            except ValueError:
                continue
            ext = base.get("ext", "png")
            img_name = f"page_{page_num}_img_{idx}.{ext}"
            img_path = output_dir / img_name
            img_path.write_bytes(base["image"])
            image_cache[xref] = str(img_path)

        for rect in rects:
            elements.append(
                ContractElement(
                    element_id=_new_id("image"),
                    element_type="image",
                    page_number=page_num,
                    bbox=BBox(x0=float(rect.x0), y0=float(rect.y0), x1=float(rect.x1), y1=float(rect.y1)),
                    order_index=-1,
                    image_path=image_cache[xref],
                    style={"source": "embedded_image"},
                )
            )
    return elements


def _dedupe_regions(elements: list[ContractElement], overlap_threshold: float = 0.75) -> list[ContractElement]:
    kept: list[ContractElement] = []
    for element in sorted(elements, key=lambda x: _bbox_area(x.bbox), reverse=True):
        if any(bbox_overlap_ratio(element.bbox, k.bbox) >= overlap_threshold for k in kept):
            continue
        kept.append(element)
    return sorted(kept, key=lambda x: (x.bbox.y0, x.bbox.x0))


def detect_signature_regions(
    page: fitz.Page,
    page_num: int,
    output_dir: Path | None = None,
    excluded_regions: list[BBox] | None = None,
) -> list[ContractElement]:
    excluded_regions = excluded_regions or []
    image = render_page_to_image(page, dpi=220).convert("RGB")
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_h, img_w = arr.shape[:2]
    page_w, page_h = float(page.rect.width), float(page.rect.height)

    output_dir_path = None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dir_path = output_dir

    elements: list[ContractElement] = []
    idx = 0
    for contour in contours:
        if len(contour) < 20:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect = w / max(h, 1)
        if area < 1600 or area > 42000:
            continue
        if not (2.5 <= aspect <= 12.0):
            continue

        stroke_density = float(cv2.countNonZero(thresh[y : y + h, x : x + w])) / float(max(area, 1))
        if stroke_density < 0.08 or stroke_density > 0.45:
            continue

        contour_area = float(cv2.contourArea(contour))
        hull = cv2.convexHull(contour)
        hull_area = max(float(cv2.contourArea(hull)), 1.0)
        solidity = contour_area / hull_area
        if solidity < 0.12 or solidity > 0.88:
            continue

        bbox = _pixel_to_page_bbox(x, y, w, h, page_w, page_h, img_w, img_h)
        if _is_bbox_in_regions(bbox, excluded_regions, threshold=0.35):
            continue

        image_path = None
        if output_dir_path is not None:
            crop = Image.fromarray(arr[y : y + h, x : x + w])
            fname = output_dir_path / f"page_{page_num}_signature_{idx}.png"
            crop.save(fname)
            image_path = str(fname)

        elements.append(
            ContractElement(
                element_id=_new_id("sig"),
                element_type="signature",
                page_number=page_num,
                bbox=bbox,
                order_index=-1,
                image_path=image_path,
                style={"source": "vision_signature"},
                metadata={"heuristic_score": round(stroke_density, 3), "solidity": round(solidity, 3)},
            )
        )
        idx += 1

    return _dedupe_regions(elements)


def detect_scanned_table_regions(page: fitz.Page, page_num: int, output_dir: Path) -> list[ContractElement]:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = render_page_to_image(page, dpi=220).convert("RGB")
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10)

    img_h, img_w = gray.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, img_w // 28), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, img_h // 30)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.add(horizontal, vertical)
    grid = cv2.dilate(grid, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    page_w, page_h = float(page.rect.width), float(page.rect.height)
    img_area = float(img_w * img_h)

    elements: list[ContractElement] = []
    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < img_area * 0.01:
            continue
        if w < img_w * 0.22 or h < img_h * 0.06:
            continue
        line_density = float(cv2.countNonZero(grid[y : y + h, x : x + w])) / max(area, 1.0)
        if line_density < 0.015:
            continue

        bbox = _pixel_to_page_bbox(x, y, w, h, page_w, page_h, img_w, img_h)
        crop_arr = arr[y : y + h, x : x + w]
        crop = Image.fromarray(crop_arr)
        path = output_dir / f"page_{page_num}_table_region_{idx}.png"
        crop.save(path)

        table_text = pytesseract.image_to_string(crop, config="--oem 3 --psm 6").strip()
        elements.append(
            ContractElement(
                element_id=_new_id("tableimg"),
                element_type="table_region",
                page_number=page_num,
                bbox=bbox,
                order_index=-1,
                text=table_text if table_text else None,
                image_path=str(path),
                style={"source": "vision_table"},
            )
        )
        idx += 1

    return _dedupe_regions(elements)


def _ocr_word_boxes_px(image: Image.Image, min_confidence: float = 25.0) -> list[tuple[int, int, int, int]]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--oem 3 --psm 6")
    boxes: list[tuple[int, int, int, int]] = []
    total = len(data.get("text", []))
    for i in range(total):
        text = str(data["text"][i] or "").strip()
        conf = _parse_conf(data["conf"][i])
        if not text or conf < min_confidence:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        boxes.append((x, y, w, h))
    return boxes


def detect_scanned_product_regions(
    page: fitz.Page,
    page_num: int,
    output_dir: Path,
    excluded_regions: list[BBox] | None = None,
) -> list[ContractElement]:
    output_dir.mkdir(parents=True, exist_ok=True)
    excluded_regions = excluded_regions or []

    image = render_page_to_image(page, dpi=220).convert("RGB")
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, binary_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    img_h, img_w = gray.shape
    page_w, page_h = float(page.rect.width), float(page.rect.height)
    img_area = float(img_w * img_h)

    ignore_mask = np.zeros_like(binary_inv)

    for x, y, w, h in _ocr_word_boxes_px(image):
        cv2.rectangle(ignore_mask, (max(0, x - 2), max(0, y - 2)), (min(img_w - 1, x + w + 2), min(img_h - 1, y + h + 2)), 255, -1)

    for region in excluded_regions:
        rx, ry, rw, rh = _page_to_pixel_bbox(region, page_w, page_h, img_w, img_h)
        cv2.rectangle(ignore_mask, (rx, ry), (min(img_w - 1, rx + rw), min(img_h - 1, ry + rh)), 255, -1)

    candidate = cv2.bitwise_and(binary_inv, cv2.bitwise_not(ignore_mask))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)

    contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    elements: list[ContractElement] = []
    idx = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = float(w * h)
        if area < img_area * 0.005 or area > img_area * 0.45:
            continue
        if w < 40 or h < 40:
            continue

        aspect = w / max(h, 1)
        if aspect < 0.35 or aspect > 5.5:
            continue

        fill_ratio = float(cv2.countNonZero(candidate[y : y + h, x : x + w])) / max(area, 1.0)
        if fill_ratio < 0.12:
            continue

        crop_gray = gray[y : y + h, x : x + w]
        texture_std = float(np.std(crop_gray))
        if texture_std < 18.0:
            continue

        bbox = _pixel_to_page_bbox(x, y, w, h, page_w, page_h, img_w, img_h)
        crop = Image.fromarray(arr[y : y + h, x : x + w])
        path = output_dir / f"page_{page_num}_product_region_{idx}.png"
        crop.save(path)

        elements.append(
            ContractElement(
                element_id=_new_id("prodimg"),
                element_type="product_image",
                page_number=page_num,
                bbox=bbox,
                order_index=-1,
                image_path=str(path),
                style={"source": "vision_product"},
            )
        )
        idx += 1

    return _dedupe_regions(elements)
