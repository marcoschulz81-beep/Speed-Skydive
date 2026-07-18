from __future__ import annotations

from app.config import BASE_DIR
from app.main import templates


def test_all_jinja_templates_parse() -> None:
    template_dir = BASE_DIR / "app" / "templates"
    for path in sorted(template_dir.glob("*.html")):
        templates.env.get_template(path.name)
