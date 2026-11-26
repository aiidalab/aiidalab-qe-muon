"""Tests of the mcif reader of the structure selector: the reason of a failure must be visible (no app needed)."""

from types import SimpleNamespace

import ipywidgets as ipw
import pytest
from _stubs import fake_modules

MINI_MCIF = """data_Fe_bcc
_cell_length_a 2.87
_cell_length_b 2.87
_cell_length_c 2.87
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_magn.name_BNS "P 1"
_space_group_magn.number_BNS 1.1
loop_
_space_group_symop_magn_operation.id
_space_group_symop_magn_operation.xyz
1 x,y,z,+1
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 Fe 0 0 0
Fe2 Fe 0.5 0.5 0.5
loop_
_atom_site_moment.label
_atom_site_moment.crystalaxis_x
_atom_site_moment.crystalaxis_y
_atom_site_moment.crystalaxis_z
Fe1 0 0 2.2
Fe2 0 0 2.2
"""


@pytest.fixture
def importer(monkeypatch):
    fake_modules(
        monkeypatch,
        "aiidalab_qe",
        "aiidalab_qe.plugins",
        "aiidalab_widgets_base",
        "aiidalab_widgets_base.data",
        "aiidalab_widgets_base.utils",
        "aiidalab_widgets_base.viewers",
    )
    from aiidalab_qe_muon.app.structure_importer import structure

    return structure


@pytest.fixture
def mcif(tmp_path):
    path = tmp_path / "Fe.mcif"
    path.write_text(MINI_MCIF)
    return path


def test_reads_the_moments_in_both_cells(importer, mcif):
    conventional = importer.get_pymatgen_from_file(str(mcif), primitive=False)
    primitive = importer.get_pymatgen_from_file(str(mcif), primitive=True)
    assert (len(conventional), len(primitive)) == (2, 1)
    assert [list(m.moment) for m in primitive.site_properties["magmom"]] == [[0.0, 0.0, 2.2]]


def test_unreadable_file_gives_the_reason(importer, tmp_path):
    bad = tmp_path / "bad.mcif"
    bad.write_text("this is not a cif file\n")
    with pytest.raises(ValueError, match=r"Could not read the file bad\.mcif: \w+") as error:
        importer.get_pymatgen_from_file(str(bad))
    assert error.value.__cause__ is not None  # the original exception is chained, not lost


def test_file_without_structure(importer, mcif, monkeypatch):
    monkeypatch.setattr(importer, "CifParser", lambda fname: SimpleNamespace(get_structures=lambda primitive: []))
    with pytest.raises(ValueError, match="No structure found in the file Fe.mcif"):
        importer.get_pymatgen_from_file(str(mcif))


def test_a_failing_parser_is_reported_not_swallowed(importer, mcif, monkeypatch):
    def broken(fname):
        raise RuntimeError("boom <b>")

    monkeypatch.setattr(importer, "CifParser", broken)
    with pytest.raises(ValueError, match="RuntimeError: boom"):
        importer.get_pymatgen_from_file(str(mcif))


def test_widget_shows_the_reason_and_clears_it(importer, monkeypatch):
    """`_read_structure` does not raise: it shows the reason, with the name of the uploaded file, until the next upload."""
    widget = SimpleNamespace(
        file_upload_subwidget1=SimpleNamespace(value=False),
        error_message=ipw.HTML(),
        _validate_and_fix_pymatgen_cell=lambda structure: "structure",
    )
    assert importer.ImportMagnetism._read_structure(widget, "my<file>.mcif", b"not a cif") is None
    message = widget.error_message.value
    assert "alert-danger" in message and "Could not read the file my&lt;file&gt;.mcif" in message
    assert "tmp" not in message  # the name of the temporary file is not shown

    assert importer.ImportMagnetism._read_structure(widget, "Fe.mcif", MINI_MCIF.encode()) == "structure"
    assert widget.error_message.value == ""
