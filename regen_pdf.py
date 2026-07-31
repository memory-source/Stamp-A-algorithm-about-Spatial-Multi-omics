import markdown
from weasyprint import HTML
import base64
from pathlib import Path

BASE = "/data/lvyongji/Assignment5"

with open(f"{BASE}/report_en.md", "r", encoding="utf-8") as f:
    md_text = f.read()

# Convert markdown to HTML body
html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

# Inline the architecture image as base64 to ensure embedding
img_path = Path(f"{BASE}/structure.png")
with open(img_path, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

# Replace file:// path with base64 data URI
html_body = html_body.replace(
    "file:///data/lvyongji/Assignment5/structure.png",
    f"data:image/png;base64,{img_b64}"
)

# Also inline other figures_v7a images
for fig in Path(f"{BASE}/code/figures_v7a").glob("*.png"):
    file_url = f"file://{fig}"
    if file_url in html_body:
        with open(fig, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        html_body = html_body.replace(file_url, f"data:image/png;base64,{b64}")

for fig in Path(f"{BASE}/code/figures").glob("*.png"):
    file_url = f"file://{fig}"
    if file_url in html_body:
        with open(fig, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        html_body = html_body.replace(file_url, f"data:image/png;base64,{b64}")

html_full = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: A4;
    margin: 2cm;
}}
body {{
    font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Helvetica Neue", Arial, sans-serif;
    font-size: 10.5pt;
    line-height: 1.55;
    color: #333;
}}
h1 {{
    font-size: 17pt;
    color: #1a1a1a;
    border-bottom: 2px solid #333;
    padding-bottom: 6px;
    margin-top: 28px;
    page-break-before: always;
}}
h1:first-of-type {{
    page-break-before: auto;
}}
h2 {{
    font-size: 13pt;
    color: #2a2a2a;
    border-bottom: 1px solid #ccc;
    padding-bottom: 3px;
    margin-top: 22px;
}}
h3 {{
    font-size: 11.5pt;
    color: #444;
    margin-top: 16px;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 10px 0;
    font-size: 9pt;
}}
th, td {{
    border: 1px solid #bbb;
    padding: 5px 7px;
    text-align: left;
}}
th {{
    background-color: #eee;
    font-weight: bold;
}}
tr:nth-child(even) {{
    background-color: #f9f9f9;
}}
code {{
    background-color: #f4f4f4;
    padding: 1px 4px;
    border-radius: 2px;
    font-family: "Courier New", monospace;
    font-size: 9.5pt;
}}
pre {{
    background-color: #f4f4f4;
    padding: 8px;
    border-radius: 4px;
    overflow-x: auto;
    font-size: 9pt;
}}
blockquote {{
    border-left: 3px solid #ccc;
    margin: 8px 0;
    padding-left: 12px;
    color: #555;
}}
img {{
    display: block;
    margin: 8px auto;
    max-width: 100%;
}}
hr {{
    border: none;
    border-top: 1px solid #ddd;
    margin: 18px 0;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

with open(f"{BASE}/report_en.html", "w", encoding="utf-8") as f:
    f.write(html_full)

HTML(string=html_full).write_pdf(f"{BASE}/report_en.pdf")
print(f"PDF generated: {BASE}/report_en.pdf")
