"""Render outputs/aud-onset-xcorr/report.md -> report.html -> report.pdf (headless Edge)."""
import pathlib
import re
import subprocess

import markdown

OUT = pathlib.Path(__file__).parent.parent.parent / "outputs" / "aud-onset-xcorr"
md = (OUT / "report.md").read_text(encoding="utf-8")
md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # drop YAML front matter
body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
css = """
@page { size: A4; margin: 14mm 13mm 14mm 13mm; }
body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9.6pt; line-height: 1.38; color: #111; }
h1 { font-size: 17pt; margin: 0 0 4pt 0; } h2 { font-size: 13pt; margin: 14pt 0 4pt; border-bottom: 1px solid #bbb; }
h3 { font-size: 11pt; margin: 10pt 0 3pt; }
table { border-collapse: collapse; margin: 6pt 0; font-size: 8.4pt; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2pt 5pt; } th { background: #f0f0f0; }
td:not(:first-child) { text-align: right; }
img { max-width: 100%; page-break-inside: avoid; margin-top: 4pt; }
p > em:only-child { font-size: 8.6pt; color: #333; }
code { font-family: Consolas, monospace; font-size: 8.6pt; background: #f5f5f5; padding: 0 2px; }
pre { background: #f5f5f5; padding: 6pt; font-size: 8pt; page-break-inside: avoid; }
li { margin: 1pt 0; }
"""
html = f"<!doctype html><html><head><meta charset='utf-8'><title>Auditory onset alignment report</title><style>{css}</style></head><body>{body}</body></html>"
(OUT / "report.html").write_text(html, encoding="utf-8")
edge = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
pdf = OUT / "report.pdf"
subprocess.run(
    [edge, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
     f"--print-to-pdf={pdf}", (OUT / "report.html").resolve().as_uri()],
    check=True, timeout=180,
)
print("wrote", pdf, pdf.stat().st_size, "bytes")
