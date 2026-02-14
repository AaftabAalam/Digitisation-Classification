from __future__ import annotations

from pathlib import Path

from contract_ai.common.io_utils import write_json
from contract_ai.common.schemas import ContractElement, ContractExtractionResult, ContractPage, ensure_dir
from contract_ai.contracts.renderer import render_html_report


class ContractExtractor:
    def __init__(self, ocr_if_low_text: bool = True, min_text_blocks: int = 2):
        self.ocr_if_low_text = ocr_if_low_text
        self.min_text_blocks = min_text_blocks

    def extract(self, pdf_path: Path, output_dir: Path) -> ContractExtractionResult:
        import fitz

        from contract_ai.contracts.extractors import (
            detect_scanned_product_regions,
            detect_scanned_table_regions,
            detect_signature_regions,
            export_page_preview,
            extract_embedded_images,
            extract_ocr_lines,
            extract_tables,
            extract_text_lines_with_style,
            page_has_large_embedded_image,
        )

        output_dir = ensure_dir(output_dir)
        assets_dir = ensure_dir(output_dir / "assets")
        page_preview_dir = ensure_dir(output_dir / "page_previews")
        doc = fitz.open(pdf_path)

        pages: list[ContractPage] = []
        all_elements: list[ContractElement] = []

        for i, page in enumerate(doc, start=1):
            pages.append(
                ContractPage(
                    page_number=i,
                    width=float(page.rect.width),
                    height=float(page.rect.height),
                    image_path=str(export_page_preview(page, i, page_preview_dir)),
                )
            )

            text_lines = extract_text_lines_with_style(page, i)

            scanned_page = self._is_scanned_page(page, text_lines, page_has_large_embedded_image(page))

            if scanned_page:
                page_elements: list[ContractElement] = []

                table_regions = detect_scanned_table_regions(page, i, assets_dir)
                page_elements.extend(table_regions)

                signatures = detect_signature_regions(
                    page,
                    i,
                    assets_dir,
                    excluded_regions=[element.bbox for element in table_regions],
                )
                page_elements.extend(signatures)

                excluded = [e.bbox for e in table_regions + signatures]
                products = detect_scanned_product_regions(page, i, assets_dir, excluded_regions=excluded)
                page_elements.extend(products)

                ocr_excluded = [element.bbox for element in table_regions + signatures + products]
                ocr_lines = extract_ocr_lines(page, i, min_confidence=35.0, excluded_regions=ocr_excluded)
                page_elements.extend(self._filter_overlapping_text(page_elements, ocr_lines))
            else:
                page_elements = []
                page_elements.extend(text_lines)
                page_elements.extend(extract_tables(pdf_path, i))
                page_elements.extend(extract_embedded_images(page, i, assets_dir))
                page_elements.extend(
                    detect_signature_regions(
                        page,
                        i,
                        assets_dir,
                        excluded_regions=[element.bbox for element in page_elements if element.element_type in {"table", "image"}],
                    )
                )

                if self.ocr_if_low_text and len(text_lines) < self.min_text_blocks:
                    ocr_lines = extract_ocr_lines(
                        page,
                        i,
                        excluded_regions=[element.bbox for element in page_elements if element.element_type in {"table", "image"}],
                    )
                    page_elements.extend(self._filter_overlapping_text(page_elements, ocr_lines))

            all_elements.extend(page_elements)

        all_elements = self._assign_global_order(all_elements)

        result = ContractExtractionResult(
            source_pdf=str(pdf_path),
            page_count=len(doc),
            pages=pages,
            elements=all_elements,
            output_dir=str(output_dir),
        )

        write_json(output_dir / "extraction.json", result.model_dump())
        render_html_report(result, output_dir / "contract_view.html")
        doc.close()
        return result

    @staticmethod
    def _is_scanned_page(page, text_lines: list[ContractElement], has_large_image: bool) -> bool:
        if has_large_image and len(text_lines) < 6:
            return True
        return len(text_lines) == 0

    @staticmethod
    def _filter_overlapping_text(
        existing_elements: list[ContractElement],
        new_text_elements: list[ContractElement],
        overlap_threshold: float = 0.82,
    ) -> list[ContractElement]:
        from contract_ai.contracts.extractors import bbox_overlap_ratio

        filtered: list[ContractElement] = []
        existing_text = [
            element
            for element in existing_elements
            if element.element_type in {"text_line", "ocr_line"}
        ]
        for candidate in new_text_elements:
            if any(bbox_overlap_ratio(candidate.bbox, current.bbox) >= overlap_threshold for current in existing_text):
                continue
            filtered.append(candidate)
        return filtered

    @staticmethod
    def _assign_global_order(elements: list[ContractElement]) -> list[ContractElement]:
        sorted_elements = sorted(
            elements,
            key=lambda e: (e.page_number, e.bbox.y0, e.bbox.x0, e.element_type),
        )
        for idx, element in enumerate(sorted_elements, start=1):
            element.order_index = idx
        return sorted_elements
