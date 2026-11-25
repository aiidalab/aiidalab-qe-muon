"""Static checks of the app texts (no app needed)."""

import re
from pathlib import Path

APP = Path(__file__).resolve().parents[1] / "src" / "aiidalab_qe_muon" / "app"


def test_plain_text_descriptions_have_no_html_entities():
    """Checkbox/Button descriptions are plain text: `&mu;` would show up literally, use the `μ` character."""
    pattern = re.compile(r"""\bdescription\s*=\s*(?:\"\"\"|\"|')[^\n]*&[a-zA-Z]+;""")
    offenders = [
        f"{path.relative_to(APP)}:{number}: {line.strip()}"
        for path in APP.rglob("*.py")
        for number, line in enumerate(path.read_text().splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, "\n".join(offenders)


def test_guides_use_the_names_of_the_app():
    for guide in (APP / "guides").glob("*.html"):
        text = guide.read_text()
        assert "Muon Spectroscopy (muons)" not in text, guide.name
        assert "mcloud: Materials Cloud" in text and "MC3D (PBEsol-v2)" in text, guide.name
        assert "Number of elements" in text, guide.name
