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


def test_spin_relaxation_guide_explains_the_undi_shell_job():
    """The tutorial with the polarization: the shell job and its code, 4 cpus-per-task, the Fast protocol."""
    text = (APP / "guides" / "1_SpinRelaxation.html").read_text()
    assert "shell job" in text and "python3-shell@localhost" in text and "aiidalab-qe-muon setup-python3" in text
    assert re.search(r"cpus-per-task</b> to <b>4</b>", text)
    assert "Select <b>Fast</b> protocol" in text and "max<sub>hdim</sub>=10<sup>5</sup>" in text
    # the plot uses 10^5 with the Fast protocol of the guide (not 5e5)
    assert "uses 5&middot;10" not in text


def test_sites_guide_has_no_polarization_to_explain():
    """The guide of the sites does not compute the polarization: no undi code to set up in it."""
    text = (APP / "guides" / "0_Sites.html").read_text()
    assert "Select <b>Fast</b> protocol" in text and "shell job" not in text
