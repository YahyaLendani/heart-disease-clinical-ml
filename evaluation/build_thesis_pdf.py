"""Compile ``thesis_draft.md`` into a publication-formatted PDF (``heart-pipeline build-thesis-pdf``).

Converts Markdown (headings, tables, lists, blockquotes, figures, inline/display LaTeX math)
into print-styled academic HTML with KaTeX math rendering and compiles it to A4 PDF via
headless Google Chrome / Chromium ``--print-to-pdf``.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
from pathlib import Path

from project_paths import PROJECT_ROOT

DEFAULT_THESIS_MD = PROJECT_ROOT / "thesis_draft.md"
DEFAULT_PDF_OUT = PROJECT_ROOT / "docs" / "CardioGraph_Clinical_AI_Thesis.pdf"


def _find_chrome() -> Path:
    candidates = [
        shutil.which("chrome"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        shutil.which("msedge"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "No Chrome/Chromium/Edge binary found for headless PDF rendering."
    )


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Temporarily replace LaTeX math spans with placeholders so Markdown escapes don't touch them."""
    store: list[str] = []

    def _sub(match: re.Match[str]) -> str:
        store.append(match.group(0))
        return f"@@MATH_{len(store) - 1}@@"

    # Protect display math $$...$$ first, then inline $...$
    text = re.sub(r"\$\$[\s\S]*?\$\$", _sub, text)
    text = re.sub(r"(?<!\\)\$(?!\s)(?:\\.|[^$\\\n])+?(?<!\s)\$", _sub, text)
    return text, store


def _restore_math(text: str, store: list[str]) -> str:
    for i, val in enumerate(store):
        text = text.replace(f"@@MATH_{i}@@", val)
    return text


def _inline_md(text: str, base_dir: Path) -> str:
    text, math_store = _protect_math(text)

    # Protect inline code spans
    code_store: list[str] = []

    def _code_sub(m: re.Match[str]) -> str:
        code_store.append(f"<code>{html.escape(m.group(1))}</code>")
        return f"@@CODE_{len(code_store) - 1}@@"

    text = re.sub(r"`([^`\n]+)`", _code_sub, text)
    text = html.escape(text)

    # Images: ![alt](src)
    def _img_sub(m: re.Match[str]) -> str:
        alt = m.group(1)
        src = m.group(2)
        if not src.startswith(("http://", "https://", "file://", "data:")):
            resolved = (base_dir / src).resolve()
            src = resolved.as_uri()
        return (
            f'<figure class="thesis-fig">'
            f'<img src="{src}" alt="{alt}" />'
            f"<figcaption>{alt}</figcaption>"
            f"</figure>"
        )

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", _img_sub, text)

    # Links: [label](url)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<span class="file-ref">\1</span>', text)

    # Bold & Italic
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    for i, code_html in enumerate(code_store):
        text = text.replace(f"@@CODE_{i}@@", code_html)
    return _restore_math(text, math_store)


def markdown_to_html_body(md_text: str, base_dir: Path) -> str:
    """Convert thesis Markdown into clean academic HTML body."""
    # Strip auto-update marker HTML comments
    md_text = re.sub(r"<!--[\s\S]*?-->", "", md_text)
    lines = md_text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Horizontal rule
        if stripped in ("---", "***", "___"):
            out.append("<hr />")
            i += 1
            continue

        # Display math block $$ ... $$
        if stripped.startswith("$$"):
            block = [line]
            if stripped == "$$" or not stripped.endswith("$$") or len(stripped) == 2:
                i += 1
                while i < n:
                    block.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        break
                    i += 1
            block_txt = "\n".join(block)
            out.append(f'<div class="math-block">\n{block_txt}\n</div>')
            i += 1
            continue

        # Fenced code block
        if stripped.startswith("```"):
            code_lines: list[str] = []
            i += 1
            while i < n and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            out.append(f"<pre><code>{html.escape(chr(10).join(code_lines))}</code></pre>")
            i += 1
            continue

        # Headings
        hm = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if hm:
            lvl = len(hm.group(1))
            if lvl == 1 and stripped.lower() == "# thesis draft":
                i += 1
                continue
            out.append(f"<h{lvl}>{_inline_md(hm.group(2).strip(), base_dir)}</h{lvl}>")
            i += 1
            continue

        # Markdown Table
        if stripped.startswith("|") and i + 1 < n and re.match(r"^\|[\s:\-|]+\|?$", lines[i + 1].strip()):
            header_cells = [c.strip() for c in stripped.strip("|").split("|")]
            i += 2
            body_rows: list[list[str]] = []
            while i < n and lines[i].strip().startswith("|"):
                body_rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            ths = "".join(f"<th>{_inline_md(c, base_dir)}</th>" for c in header_cells)
            trs = "".join(
                "<tr>" + "".join(f"<td>{_inline_md(c, base_dir)}</td>" for c in row) + "</tr>"
                for row in body_rows
            )
            out.append(
                f'<div class="table-wrap"><table><thead><tr>{ths}</tr></thead>'
                f"<tbody>{trs}</tbody></table></div>"
            )
            continue

        # Blockquote
        if stripped.startswith(">"):
            q_lines = []
            while i < n and lines[i].strip().startswith(">"):
                q_lines.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append(f"<blockquote>{_inline_md(' '.join(q_lines), base_dir)}</blockquote>")
            continue

        # Unordered list
        if re.match(r"^[-*]\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                item_txt = re.sub(r"^\s*[-*]\s+", "", lines[i]).strip()
                items.append(f"<li>{_inline_md(item_txt, base_dir)}</li>")
                i += 1
            out.append(f"<ul>{''.join(items)}</ul>")
            continue

        # Ordered list
        if re.match(r"^\d+\.\s+", stripped):
            items = []
            while i < n and re.match(r"^\s*\d+\.\s+", lines[i]):
                item_txt = re.sub(r"^\s*\d+\.\s+", "", lines[i]).strip()
                # Gather indented continuation lines
                while i + 1 < n and lines[i + 1].startswith("   ") and lines[i + 1].strip():
                    i += 1
                    item_txt += " " + lines[i].strip()
                items.append(f"<li>{_inline_md(item_txt, base_dir)}</li>")
                i += 1
            out.append(f"<ol>{''.join(items)}</ol>")
            continue

        # Standalone image paragraph
        if re.match(r"^!\[([^\]]*)\]\(([^)]+)\)$", stripped):
            out.append(_inline_md(stripped, base_dir))
            i += 1
            continue

        # Regular paragraph
        p_lines = [stripped]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not lines[i].strip().startswith(("#", "|", ">", "```", "$$", "---", "- ", "* "))
            and not re.match(r"^\d+\.\s+", lines[i].strip())
        ):
            p_lines.append(lines[i].strip())
            i += 1
        out.append(f"<p>{_inline_md(' '.join(p_lines), base_dir)}</p>")

    return "\n".join(out)


def build_thesis_pdf(
    thesis_md: str | Path = DEFAULT_THESIS_MD,
    output_pdf: str | Path = DEFAULT_PDF_OUT,
) -> Path:
    """Render ``thesis_draft.md`` to a print-ready A4 PDF using KaTeX + headless Chrome."""
    md_path = Path(thesis_md).resolve()
    pdf_path = Path(output_pdf).resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    body_html = markdown_to_html_body(md_path.read_text(encoding="utf-8"), md_path.parent)
    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Graph-Augmented Transformer for Interpretable Clinical Risk Prediction</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
      {{left: '$$', right: '$$', display: true}},
      {{left: '$', right: '$', display: false}}
    ], throwOnError: false}});"></script>
  <style>
    @page {{
      size: A4;
      margin: 20mm 18mm 22mm 18mm;
      @bottom-right {{ content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #5a6e82; }}
      @top-left {{ content: "Graph-Augmented Clinical Transformer — Thesis Monograph"; font-size: 8.5pt; color: #5a6e82; }}
    }}
    @page :first {{ @top-left {{ content: none; }} @bottom-right {{ content: none; }} }}
    body {{
      font-family: "Georgia", "Times New Roman", serif;
      font-size: 10.2pt;
      line-height: 1.52;
      color: #132230;
      max-width: 100%;
      margin: 0;
    }}
    .cover {{
      page-break-after: always;
      min-height: 235mm;
      display: flex;
      flex-direction: column;
      justify-content: center;
      border-left: 5px solid #0f5c6e;
      padding: 0 28px;
    }}
    .cover-kicker {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 10pt;
      font-weight: 700;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: #0f5c6e;
      margin-bottom: 18px;
    }}
    .cover h1 {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 26pt;
      line-height: 1.18;
      color: #0b1925;
      margin: 0 0 18px 0;
      border: none;
    }}
    .cover-sub {{
      font-size: 12pt;
      color: #3a4f63;
      line-height: 1.5;
      margin-bottom: 34px;
    }}
    .cover-meta {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 10pt;
      color: #24384a;
      line-height: 1.8;
      border-top: 1px solid #d5dfe8;
      padding-top: 18px;
    }}
    .cover-meta a {{ color: #0f5c6e; text-decoration: none; }}
    h2 {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 15pt;
      color: #0f5c6e;
      border-bottom: 1.5px solid #d5e3e8;
      padding-bottom: 4px;
      margin: 22px 0 10px 0;
      break-after: avoid;
    }}
    h3 {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 11.8pt;
      color: #132230;
      margin: 16px 0 6px 0;
      break-after: avoid;
    }}
    h4 {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 10.5pt;
      color: #2a4154;
      margin: 12px 0 4px 0;
      break-after: avoid;
    }}
    p {{ margin: 0 0 8.5px 0; text-align: justify; }}
    ul, ol {{ margin: 4px 0 10px 0; padding-left: 22px; }}
    li {{ margin-bottom: 4px; }}
    code {{
      font-family: "Consolas", "Courier New", monospace;
      font-size: 8.8pt;
      background: #f0f5f8;
      padding: 1px 4px;
      border-radius: 3px;
      color: #0f4c5c;
    }}
    pre {{
      background: #f5f8fa;
      border: 1px solid #dce5ec;
      padding: 10px 12px;
      border-radius: 5px;
      overflow-x: auto;
      font-size: 8.2pt;
      break-inside: avoid;
    }}
    pre code {{ background: none; padding: 0; }}
    blockquote {{
      border-left: 3px solid #0f5c6e;
      background: #f4f9fa;
      margin: 10px 0;
      padding: 8px 14px;
      color: #22384a;
      break-inside: avoid;
    }}
    .table-wrap {{ margin: 12px 0; break-inside: avoid; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 8pt;
      font-variant-numeric: tabular-nums;
    }}
    th {{
      background: #0f5c6e;
      color: #ffffff;
      text-align: left;
      padding: 5.5px 6px;
      font-weight: 600;
    }}
    td {{
      border-bottom: 1px solid #dde6ec;
      padding: 4.5px 6px;
      vertical-align: top;
    }}
    tr:nth-child(even) td {{ background: #f8fbfc; }}
    .thesis-fig {{
      margin: 14px auto;
      text-align: center;
      break-inside: avoid;
    }}
    .thesis-fig img {{
      max-width: 96%;
      max-height: 118mm;
      object-fit: contain;
      border: 1px solid #e2eaf0;
      border-radius: 4px;
      padding: 4px;
      background: #ffffff;
    }}
    .thesis-fig figcaption {{
      font-family: "Segoe UI", Arial, sans-serif;
      font-size: 8.5pt;
      color: #4b6073;
      margin-top: 5px;
      font-style: italic;
    }}
    .math-block {{ margin: 10px 0; text-align: center; break-inside: avoid; }}
    hr {{ border: none; border-top: 1px solid #d5dfe8; margin: 16px 0; }}
    a {{ color: #0f5c6e; text-decoration: none; }}
  </style>
</head>
<body>
  <section class="cover">
    <div class="cover-kicker">Computer Science Research Thesis &middot; Clinical Artificial Intelligence</div>
    <h1>Graph-Augmented Transformer for Interpretable Clinical Risk Prediction</h1>
    <p class="cover-sub">
      A Zero-Pickle Two-Stage Clinical Decision Support Framework with Periodic Numerical Embeddings,
      Inductive Cohort Graph Attention, Mondrian &amp; Covariate-Shift Weighted Conformal Triage,
      Real 12-Lead ECG Multimodal Fusion, and HL7 FHIR R4 Interoperability.
    </p>
    <div class="cover-meta">
      <strong>Author:</strong> Yahya Lendani (Warsaw, Poland)<br />
      <strong>Year:</strong> 2026 &middot; Research Prototype (Not FDA-Cleared or CE-Marked SaMD)<br />
      <strong>Live 2-Stage CDS Demo:</strong> <a href="https://cardiograph-cds.onrender.com">https://cardiograph-cds.onrender.com</a><br />
      <strong>Interactive Defense Slides:</strong> <a href="https://yahyalendani.me/cardiograph-slides/">https://yahyalendani.me/cardiograph-slides/</a><br />
      <strong>Source &amp; Reproducibility Suite:</strong> <a href="https://github.com/YahyaLendani/heart-disease-clinical-ml">github.com/YahyaLendani/heart-disease-clinical-ml</a>
    </div>
  </section>
  <main>
    {body_html}
  </main>
</body>
</html>
"""

    tmp_html = pdf_path.with_suffix(".print.html")
    tmp_html.write_text(full_html, encoding="utf-8")

    chrome = _find_chrome()
    cmd = [
        str(chrome),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--virtual-time-budget=12000",
        "--no-pdf-header-footer",
        f"--print-to-pdf={pdf_path}",
        tmp_html.as_uri(),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    tmp_html.unlink(missing_ok=True)
    if res.returncode != 0 or not pdf_path.exists() or pdf_path.stat().st_size < 10_000:
        raise RuntimeError(
            f"Headless Chrome PDF build failed (code={res.returncode}): {res.stderr[:500]}"
        )
    print(f"Built thesis PDF -> {pdf_path} ({pdf_path.stat().st_size / 1024:.1f} KB)")
    return pdf_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compile thesis_draft.md into publication PDF.")
    parser.add_argument("--input", type=str, default=str(DEFAULT_THESIS_MD))
    parser.add_argument("--output", type=str, default=str(DEFAULT_PDF_OUT))
    args = parser.parse_args(argv)
    build_thesis_pdf(thesis_md=args.input, output_pdf=args.output)


if __name__ == "__main__":
    main()
