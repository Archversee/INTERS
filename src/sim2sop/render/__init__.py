"""Render SOPs to HTML (Jinja2) and PDF (WeasyPrint, optional)."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from sim2sop import config
from sim2sop.schema import SOP

_env = Environment(loader=PackageLoader("sim2sop.render", "templates"),
                   autoescape=select_autoescape(["html", "j2"]))


def to_html(sop: SOP) -> str:
    return _env.get_template("sop.html.j2").render(sop=sop)


def write_html(sop: SOP, path: Path | None = None) -> Path:
    path = path or config.SOPS_DIR / sop.procedure / f"v{sop.version}.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_html(sop), encoding="utf-8")
    return path


def write_pdf(sop: SOP, path: Path | None = None) -> Path:
    try:
        from weasyprint import HTML
    except (ImportError, OSError) as e:  # OSError: GTK/Pango missing on Windows
        raise RuntimeError("PDF export needs WeasyPrint: pip install -e '.[pdf]' "
                           "(on Windows also install the GTK runtime)") from e
    path = path or config.SOPS_DIR / sop.procedure / f"v{sop.version}.pdf"
    HTML(string=to_html(sop)).write_pdf(str(path))
    return path
