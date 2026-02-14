from __future__ import annotations

from pathlib import Path

from jinja2 import Template

from contract_ai.common.schemas import ContractExtractionResult

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Contract Extraction View</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --ink: #1f2937;
      --panel: #ffffff;
      --line: #d3d9e2;
      --muted: #556070;
      --accent: #1357d5;
    }
    body {
      margin: 24px;
      color: var(--ink);
      background: linear-gradient(180deg, #f8fbff 0%, #f7f8fb 100%);
      font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    }
    h1, h2 {
      margin: 0;
    }
    .meta {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .section {
      margin-top: 20px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 8px 30px rgba(16, 36, 66, 0.05);
    }
    .page-wrapper {
      margin-top: 18px;
      border: 1px dashed #c6cfdd;
      border-radius: 8px;
      padding: 14px;
      background: #fbfcff;
    }
    .compare-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      align-items: start;
    }
    .panel-label {
      font-size: 12px;
      margin-bottom: 6px;
      color: #334765;
      font-weight: 700;
      letter-spacing: 0.03em;
      text-transform: uppercase;
    }
    .original-panel {
      background: #fff;
      border: 1px solid #bac6db;
      box-shadow: 0 8px 24px rgba(25, 42, 70, 0.11);
      overflow: hidden;
    }
    .original-panel img {
      display: block;
      width: 100%;
      height: auto;
    }
    .page-label {
      font-weight: 700;
      color: #1e3658;
      margin-bottom: 10px;
    }
    .page-canvas {
      position: relative;
      background: #fff;
      border: 1px solid #bac6db;
      box-shadow: 0 8px 24px rgba(25, 42, 70, 0.11);
      transform-origin: top left;
      overflow: hidden;
    }
    .text-layer {
      position: absolute;
      white-space: pre-wrap;
      line-height: 1.18;
      color: #111827;
      z-index: 4;
    }
    .asset-layer {
      position: absolute;
      object-fit: contain;
      border: 1px solid #b4bfd1;
      z-index: 2;
      background: #fff;
    }
    .table-layer {
      position: absolute;
      border: 1px solid #293449;
      border-collapse: collapse;
      font-size: 11px;
      z-index: 3;
      background: #fff;
    }
    .table-layer td {
      border: 1px solid #293449;
      padding: 2px 3px;
      vertical-align: top;
    }
    .badge {
      position: absolute;
      right: 4px;
      top: 4px;
      padding: 2px 5px;
      font-size: 10px;
      color: #fff;
      background: #384e72;
      border-radius: 4px;
      z-index: 5;
    }
    table.summary {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 10px;
    }
    table.summary th, table.summary td {
      border: 1px solid #cfd6e4;
      padding: 6px;
      text-align: left;
      vertical-align: top;
    }
    table.summary th {
      background: #eff4ff;
      color: #203758;
    }
    .mono {
      font-family: Menlo, Consolas, monospace;
      font-size: 11px;
    }
  </style>
</head>
<body>
  <h1>Contract Reconstruction Report</h1>
  <div class="meta">
    Source: {{ source_pdf }}<br />
    Pages: {{ page_count }} | Elements: {{ elements|length }} | Scale: {{ render_scale }}x
  </div>

  <div class="section">
    <h2>Page-By-Page Reconstructed Layout</h2>
    {% for page in pages %}
      <div class="page-wrapper">
        <div class="page-label">Page {{ page.page_number }} ({{ page.width }} x {{ page.height }})</div>
        <div class="compare-grid">
          <div>
            <div class="panel-label">Original</div>
            <div class="original-panel" style="width: {{ page.width * render_scale }}px;">
              {% if page.image_path %}
                <img src="{{ page.image_path }}" alt="Original page {{ page.page_number }}" />
              {% endif %}
            </div>
          </div>
          <div>
            <div class="panel-label">Reconstructed</div>
            <div class="page-canvas" style="width: {{ page.width * render_scale }}px; height: {{ page.height * render_scale }}px;">
              {% for el in page_elements.get(page.page_number, []) %}
                {% set left = el.bbox.x0 * render_scale %}
                {% set top = el.bbox.y0 * render_scale %}
                {% set width = (el.bbox.x1 - el.bbox.x0) * render_scale %}
                {% set height = (el.bbox.y1 - el.bbox.y0) * render_scale %}

                {% if el.element_type in ["text_line", "ocr_line"] %}
                  <div
                    class="text-layer"
                    style="
                      left: {{ left }}px;
                      top: {{ top }}px;
                      width: {{ width }}px;
                      min-height: {{ height }}px;
                      font-size: {{ (el.style.font_size or 10.0) * render_scale * 0.95 }}px;
                      text-align: {{ el.style.alignment or 'left' }};
                      font-weight: {{ 700 if el.style.bold else 400 }};
                      font-style: {{ 'italic' if el.style.italic else 'normal' }};
                      text-decoration: {{ 'underline' if el.style.underline else 'none' }};
                    "
                  >{{ el.text | e }}</div>
                {% elif el.element_type == "table" and el.table_data %}
                  <table class="table-layer" style="left: {{ left }}px; top: {{ top }}px; width: {{ width }}px;">
                    {% for row in el.table_data %}
                      <tr>
                        {% for cell in row %}
                          <td>{{ cell | e }}</td>
                        {% endfor %}
                      </tr>
                    {% endfor %}
                  </table>
                {% elif el.image_path %}
                  <img class="asset-layer" src="{{ el.image_path }}" style="left: {{ left }}px; top: {{ top }}px; width: {{ width }}px; height: {{ height }}px;" />
                  <div class="badge" style="left: {{ left + 4 }}px; top: {{ top + 4 }}px; right: auto;">{{ el.element_type }}</div>
                {% endif %}
              {% endfor %}
            </div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>

  <div class="section">
    <h2>Extracted Element Summary</h2>
    <table class="summary">
      <thead>
        <tr>
          <th>Order</th>
          <th>Page</th>
          <th>Type</th>
          <th>BBox</th>
          <th>Preview</th>
          <th>Asset</th>
        </tr>
      </thead>
      <tbody>
        {% for el in elements %}
          <tr>
            <td>{{ el.order_index }}</td>
            <td>{{ el.page_number }}</td>
            <td>{{ el.element_type }}</td>
            <td class="mono">{{ el.bbox.x0 }},{{ el.bbox.y0 }},{{ el.bbox.x1 }},{{ el.bbox.y1 }}</td>
            <td>{{ (el.text or '')[:220] | e }}</td>
            <td>{% if el.image_path %}<span class="mono">{{ el.image_path }}</span>{% endif %}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


def _relative_asset_path(asset_path: str, html_path: Path) -> str:
    path = Path(asset_path)
    try:
        return str(path.resolve().relative_to(html_path.parent.resolve()))
    except ValueError:
        return str(path)


def render_html_report(result: ContractExtractionResult, output_path: Path) -> None:
    elements = [element.model_dump() for element in result.elements]
    for element in elements:
        if element.get("image_path"):
            element["image_path"] = _relative_asset_path(element["image_path"], output_path)

    page_elements: dict[int, list[dict]] = {}
    for element in elements:
        page_elements.setdefault(element["page_number"], []).append(element)

    for key in page_elements:
        page_elements[key] = sorted(
            page_elements[key],
            key=lambda item: (item["bbox"]["y0"], item["bbox"]["x0"], item["order_index"]),
        )

    pages = [page.model_dump() for page in result.pages]
    for page in pages:
        if page.get("image_path"):
            page["image_path"] = _relative_asset_path(page["image_path"], output_path)

    template = Template(HTML_TEMPLATE)
    html = template.render(
        source_pdf=result.source_pdf,
        page_count=result.page_count,
        pages=pages,
        elements=elements,
        page_elements=page_elements,
        render_scale=1.0,
    )
    output_path.write_text(html, encoding="utf-8")
