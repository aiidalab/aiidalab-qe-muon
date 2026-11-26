"""Tests of the collinearization of the magnetic moments shown in the muon settings (no app needed)."""

import re

import pytest
from _stubs import fake_modules
from aiida import orm

from aiidalab_qe_muon.utils.magnetism import collinearize_magmoms, magmoms_table_html


@pytest.fixture(autouse=True)
def _pylocfield(monkeypatch):
    """`aiida-muon` imports `pylocfield` (dipolar fields), which these tests do not use: stand in for it if missing."""
    fake_modules(monkeypatch, "pylocfield", "pylocfield.ewald", "pylocfield.realspace")


@pytest.fixture
def mno_like():
    """Two Mn (antiferromagnetic) and two O."""
    structure = orm.StructureData(cell=[[4.4, 0, 0], [0, 4.4, 0], [0, 0, 8.8]])
    for position, symbol in [
        ((0, 0, 0), "Mn"),
        ((0, 0, 4.4), "Mn"),
        ((2.2, 2.2, 2.2), "O"),
        ((2.2, 2.2, 6.6), "O"),
    ]:
        structure.append_atom(position=position, symbols=symbol)
    return structure


def test_antiferromagnetic_gives_two_kinds(mno_like):
    result = collinearize_magmoms(
        mno_like, [[0, 0, 4.5], [0, 0, -4.5], [0, 0, 0], [0, 0, 0]]
    )
    rows = result["rows"]
    assert [row["site"] for row in rows] == [0, 1, 2, 3]
    assert [row["element"] for row in rows] == ["Mn", "Mn", "O", "O"]
    # two distinct Mn kinds, with opposite starting magnetization; O is not magnetic.
    assert rows[0]["kind"] != rows[1]["kind"]
    assert {row["start_mag"] for row in rows[:2]} == {0.5, -0.5}
    assert [row["kind"] for row in rows[2:]] == ["O", "O"]
    assert [row["start_mag"] for row in rows[2:]] == [0.0, 0.0]
    assert sorted(result["start_mag"].values()) == [-0.5, 0.5]


def test_moments_are_made_collinear_along_z(mno_like):
    """Moments along x become `[0, 0, mz]`, keeping the size and the relative sign."""
    rows = collinearize_magmoms(
        mno_like, [[4.5, 0, 0], [-4.5, 0, 0], [0, 0, 0], [0, 0, 0]]
    )["rows"]
    for row in rows[:2]:
        assert row["magmom"][:2] == pytest.approx([0.0, 0.0])
        assert abs(row["magmom"][2]) == pytest.approx(4.5)
    assert rows[0]["magmom"][2] == pytest.approx(-rows[1]["magmom"][2])


def test_ferromagnetic_has_one_kind(mno_like):
    rows = collinearize_magmoms(
        mno_like, [[0, 0, 4.5], [0, 0, 4.5], [0, 0, 0], [0, 0, 0]]
    )["rows"]
    assert rows[0]["kind"] == rows[1]["kind"]
    assert rows[0]["start_mag"] == rows[1]["start_mag"] == 0.5


def test_full_moment_when_not_half(mno_like):
    rows = collinearize_magmoms(
        mno_like, [[0, 0, 4.5], [0, 0, -4.5], [0, 0, 0], [0, 0, 0]], half=False
    )["rows"]
    assert {row["start_mag"] for row in rows[:2]} == {4.5, -4.5}


def test_wrong_number_of_moments(mno_like):
    with pytest.raises(ValueError, match="magnetic moments"):
        collinearize_magmoms(mno_like, [[0, 0, 1.0]])


def test_helper_does_not_create_provenance(mno_like):
    """The structure is not stored and no calculation node is created."""
    before = orm.QueryBuilder().append(orm.CalcFunctionNode).count()
    collinearize_magmoms(mno_like, [[0, 0, 4.5], [0, 0, -4.5], [0, 0, 0], [0, 0, 0]])
    assert orm.QueryBuilder().append(orm.CalcFunctionNode).count() == before
    assert not mno_like.is_stored


# ---- the table shown in the settings panel

ROWS = [
    {"site": 0, "element": "Mn", "kind": "Mn1", "magmom": [0.0, 0.0, 4.5], "start_mag": 0.5},
    {"site": 1, "element": "Mn", "kind": "Mn2", "magmom": [0.0, 0.0, -4.5], "start_mag": -0.5},
]


def _cells(html, tag):
    return [re.sub(r"<[^>]+>", "", c) for c in re.findall(rf"<{tag}[^>]*>(.*?)</{tag}>", html)]


def test_table_has_one_row_per_site_with_the_collinear_columns_last():
    html = magmoms_table_html(["Mn", "Mn"], [[0, 0, 4.5], [0, 0, -4.5]], ROWS)
    # one table, the new kind before the collinear moment and the starting magnetization
    assert html.count("<table") == 1
    assert _cells(html, "th")[2:] == [
        "Site", "Kind", "mx (μB)", "my", "mz", "New kind", "mz (μB)", "Starting magnetization"
    ]
    rows = [_cells(row, "td") for row in re.findall(r"<tr>(?:<td.*?</td>)+</tr>", html)]
    assert rows == [
        ["0", "Mn", "0.000", "0.000", "4.500", "Mn1", "4.500", "+0.5"],
        ["1", "Mn", "0.000", "0.000", "-4.500", "Mn2", "-4.500", "-0.5"],
    ]


def test_starting_magnetization_is_bold():
    html = magmoms_table_html(["Mn", "Mn"], [[0, 0, 4.5], [0, 0, -4.5]], ROWS)
    assert re.findall(r"<b>([^<]*)</b>", html) == ["+0.5", "-0.5"]


def test_table_without_collinear_moments_shows_the_reason():
    html = magmoms_table_html(["Mn"], [[0, 0, 4.5]], [], error="ValueError: <boom>")
    row = _cells(re.findall(r"<tr>(?:<td.*?</td>)+</tr>", html)[0], "td")
    assert row == ["0", "Mn", "0.000", "0.000", "4.500", "–", "–", "–"]
    assert "could not be computed: ValueError: &lt;boom&gt;" in html
