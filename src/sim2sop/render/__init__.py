"""Render guidelines to HTML (Jinja2) and PDF (WeasyPrint, optional)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from sim2sop.schema import Guidelines

_env = Environment(loader=PackageLoader("sim2sop.render", "templates"),
                   autoescape=select_autoescape(["html", "j2"]))


def to_html(doc: Guidelines) -> str:
    return _env.get_template("guidelines.html.j2").render(doc=doc)


def write_html(doc: Guidelines, path: Path | None = None) -> Path:
    from sim2sop.guidelines_store import html_path
    path = path or html_path(doc)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(doc), encoding="utf-8")
    return path


def write_pdf(doc: Guidelines, path: Path | None = None) -> Path:
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:  # OSError: GTK/Pango missing on Windows
        raise RuntimeError("PDF export needs WeasyPrint: pip install -e '.[pdf]' "
                           "(on Windows also install the GTK runtime)") from e
    from sim2sop.guidelines_store import html_path
    path = path or html_path(doc).with_suffix(".pdf")
    HTML(string=to_html(doc)).write_pdf(str(path))
    return path
