"""Convert FINDINGS.md to PDF with embedded figures using WeasyPrint."""

import os
import markdown
from weasyprint import HTML

_script_dir = os.path.dirname(os.path.abspath(__file__))


def main():
    md_path = os.path.join(_script_dir, "FINDINGS.md")
    with open(md_path) as f:
        md_text = f.read()

    # Convert relative image paths to absolute file:// URIs so WeasyPrint
    # can resolve them regardless of working directory.
    figures_dir = os.path.join(_script_dir, "figures")
    md_text = md_text.replace("](figures/", f"]({figures_dir}/")

    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code"],
    )

    html_doc = f"""\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {{
        size: letter;
        margin: 1in 0.9in;
    }}
    body {{
        font-family: system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.5;
        color: #1a1a1a;
        max-width: 100%;
    }}
    h1 {{
        font-size: 16pt;
        margin-top: 0;
        margin-bottom: 0.4em;
        color: #111;
    }}
    h2 {{
        font-size: 13pt;
        margin-top: 1.4em;
        margin-bottom: 0.4em;
        color: #222;
        border-bottom: 1px solid #ddd;
        padding-bottom: 0.2em;
    }}
    p {{
        margin: 0.5em 0;
    }}
    img {{
        max-width: 100%;
        display: block;
        margin: 0.8em auto;
    }}
    table {{
        border-collapse: collapse;
        margin: 0.8em 0;
        font-size: 10pt;
        width: 100%;
    }}
    th, td {{
        border: 1px solid #ccc;
        padding: 0.35em 0.6em;
        text-align: left;
    }}
    th {{
        background-color: #f5f5f5;
        font-weight: 600;
    }}
    code {{
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 0.9em;
        background: #f4f4f4;
        padding: 0.1em 0.3em;
        border-radius: 3px;
    }}
    pre {{
        background: #f4f4f4;
        padding: 0.7em 1em;
        border-radius: 4px;
        overflow-x: auto;
        font-size: 9.5pt;
    }}
    pre code {{
        background: none;
        padding: 0;
    }}
    ul {{
        margin: 0.4em 0;
        padding-left: 1.5em;
    }}
    li {{
        margin: 0.3em 0;
    }}
    a {{
        color: #0366d6;
        text-decoration: none;
    }}
</style>
</head>
<body>
{html_body}
</body>
</html>
"""

    output_path = os.path.join(_script_dir, "FINDINGS.pdf")
    HTML(string=html_doc, base_url=_script_dir).write_pdf(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
