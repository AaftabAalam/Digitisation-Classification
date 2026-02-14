# Digitisation-Classification
A project based on contract digitisation and product classification developed as an alternate version of my original work just to check the working and implementation of other models and logics as my major experiements. 
# Contract Digitization AI

End-to-end AI system for:
1. Extracting **full contract content** from selectable and non-selectable PDFs in page order (formatted text lines, tables, product/image regions, signature regions).
2. Classifying product images by appearance (fridge, HVAC, etc.).
3. Auto-retraining classification models from low-confidence/misclassified samples and promoting updated model artifacts.

## Core Features
- Hybrid PDF extraction:
  - Selectable PDF line and span extraction with style metadata (bold/italic/underline heuristic/alignment)
  - OCR line extraction for scanned pages (Tesseract) with page-coordinate mapping
  - Region-first scanned extraction (table/product/signature) and OCR masking to avoid noisy text from non-text zones
  - Table extraction for selectable PDFs (pdfplumber) and table-region detection for scanned PDFs
  - Product/image-region and signature-region detection on scanned pages (no full-page scan asset export)
- Structured output preserving sequence:
  - JSON with page size + bounding boxes + reading order index + style metadata
  - Reconstructed HTML view that places extracted elements back at original positions per page
  - Side-by-side page comparison (original page preview vs reconstructed layout)
- Product classification:
  - Zero-shot CLIP baseline for immediate usage
  - Trainable embedding classifier for custom classes
- Auto-retraining loop:
  - Low-confidence samples automatically queued
  - Labeled feedback consumed by retraining job
  - New model version promoted automatically

## Project Layout
- `/src/contract_ai/contracts` - Contract extraction pipeline
- `/src/contract_ai/classification` - Product image classifier + training
- `/src/contract_ai/retraining` - Feedback ingest + auto model update
- `/src/contract_ai/api` - FastAPI service
- `/scripts` - Utilities for running retrain workers / data jobs
- `/data` - Runtime data, models, exports, feedback

## Requirements
- Python 3.10+
- Tesseract OCR installed locally:
  - macOS: `brew install tesseract`
  - Ubuntu: `sudo apt install tesseract-ocr`
- Ghostscript + Tk (if using Camelot lattice mode)

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## CLI Usage
### 1) Extract contract PDF
```bash
contract-ai extract-contract --input data/incoming/sample_contract.pdf --output-dir data/exports/sample_contract
```
Outputs:
- `extraction.json` (ordered elements with bboxes, text styles, and page metadata)
- `contract_view.html` (page reconstruction + summary for structure verification)
- `assets/` (cropped region assets such as table/product/signature, not full-page scan image)

### 2) Classify product image
```bash
contract-ai classify-image --input data/incoming/product.jpg --labels fridge,hvac,washing_machine,microwave
```

### 3) Provide feedback for wrong prediction
```bash
contract-ai add-feedback --input data/incoming/product.jpg --true-label hvac
```

### 4) Trigger retraining
```bash
contract-ai retrain
```

## API Usage
Start API:
```bash
uvicorn contract_ai.api.main:app --reload
```

Key endpoints:
- `POST /contracts/extract`
- `POST /classification/predict`
- `POST /feedback/classification`
- `POST /retrain/run`

## How sequence verification works
Each extracted element includes:
- `page_number`
- `bbox` (x0,y0,x1,y1)
- `element_type` (`text_line`, `ocr_line`, `table`, `table_region`, `image`, `product_image`, `signature`)
- `order_index`

Sorting by `(page_number, bbox.y0, bbox.x0)` reconstructs reading order and helps verify positional match against original contract.

## Auto-adaptation behavior
- If prediction confidence is below threshold, sample is automatically added to review queue.
- When enough reviewed samples exist, retraining runs and updates active model artifact (`data/models/classifier/current`).

## Notes
- Contract extraction uses deterministic layout ordering and is designed for auditability.
- For domain-specific contracts, accuracy improves significantly after feedback-driven retraining.
