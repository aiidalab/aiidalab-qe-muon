"""Tests of how the app shows and exports the identifiers of the muon sites.

A site is identified by its `supercell_index` (the index of the tables, and the first column of the csv files) and by
its `atom_index_in_unit_cell` (the 0-based index of its muon in the unit cell with all the sites): see
`aiida_muon.utils.export_findmuon.produce_muonic_dataframe`.
"""

import base64
import csv
import io
import sys
import types
import zipfile

import pandas as pd
import pytest
from jinja2 import Environment

try:  # the app needs only `Model` from aiidalab-qe: use a stand-in where it is not installed.
    import aiidalab_qe.common.mvc  # noqa: F401
except ImportError:
    import traitlets as tl

    for name in ("aiidalab_qe", "aiidalab_qe.common", "aiidalab_qe.common.mvc"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules["aiidalab_qe.common.mvc"].Model = type("Model", (tl.HasTraits,), {})

from pathlib import Path

from aiidalab_qe_muon.utils.data import dictionary_of_names_for_html

APP = Path(__file__).resolve().parents[1] / "src" / "aiidalab_qe_muon" / "app"
LEGEND_TEMPLATE = (APP / "static" / "templates" / "table_legend.html.j2").read_text()


def _load_findmuon_model():
    """The model; loaded from its file if the app package cannot be imported (it needs all of aiidalab-qe)."""
    try:
        from aiidalab_qe_muon.app.results.sub_mvc.findmuonmodel import FindMuonModel
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location("findmuonmodel", APP / "results" / "sub_mvc" / "findmuonmodel.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        FindMuonModel = module.FindMuonModel
    return FindMuonModel


FindMuonModel = _load_findmuon_model()

COLUMNS = [
    "structure_id_pk",
    "label",
    "delta_E",
    "tot_energy",
    "muon_position_cc",
    "B_T",
    "Bdip",
    "B_T_norm",
    "Bdip_norm",
    "B_hf_norm",
    "atom_index_in_unit_cell",
]
DEPRECATED = ("muon_index", "muon_index_global_unitcell")


def make_table(index=(3, 0)):
    """A table as exported: the supercell index is the (named) index and a column; the sites are in order of energy."""
    index = pd.Index(list(index), name="supercell_index")
    return pd.DataFrame(
        {
            "structure_id_pk": [1000 + i for i in index],
            "label": ["A", "B"],
            "delta_E": [0, 5],
            "tot_energy": [-100.0, -95.0],
            "muon_position_cc": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "atom_index_in_unit_cell": [8, 9],
            "supercell_index": list(index),
        },
        index=index,
    )


@pytest.fixture
def model():
    model = FindMuonModel()
    model.findmuon_data = {"table": make_table(), "table_all": make_table()}
    return model


@pytest.mark.parametrize("advanced", [False, True])
def test_table_columns_shown(model, advanced):
    model.advanced_table = advanced
    model._generate_table_data()
    header = model.table_data[0]
    assert dictionary_of_names_for_html["label"] in header
    # the supercell index is not shown (it is the index of the table); the atom index only in the advanced mode.
    assert (dictionary_of_names_for_html["atom_index_in_unit_cell"] in header) is advanced
    assert (dictionary_of_names_for_html["tot_energy"] in header) is advanced
    assert len(model.table_data) == 1 + len(model.findmuon_data["table"])
    assert all(len(row) == len(header) for row in model.table_data[1:])


def test_every_exported_column_has_a_name():
    for column in COLUMNS:
        assert column in dictionary_of_names_for_html
    for column in DEPRECATED:
        assert column not in dictionary_of_names_for_html


def test_csv_files_first_column_is_the_supercell_index(model):
    files_dict = {
        "table": model.findmuon_data["table"],
        "table_all": model.findmuon_data["table_all"],
        "structures": {},
        "distortions": {},
        "readme": "readme",
    }
    payload = FindMuonModel.produce_bitestream(files_dict)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        for name in ("Summary_table.csv", "Summary_table_before_clustering.csv"):
            rows = list(csv.reader(io.StringIO(archive.read(name).decode())))
            header = rows[0]
            assert header[0] == "supercell_index"
            assert "supercell_index" not in header[1:]  # not repeated
            assert "atom_index_in_unit_cell" in header
            assert [row[0] for row in rows[1:]] == ["3", "0"]
            assert [row[header.index("atom_index_in_unit_cell")] for row in rows[1:]] == ["8", "9"]
        assert "README.txt" in " ".join(archive.namelist())


@pytest.mark.parametrize("download_mode", [False, True])
@pytest.mark.parametrize("advanced_table", [False, True])
@pytest.mark.parametrize("B_fields", [False, True])
def test_legend_renders(download_mode, advanced_table, B_fields):
    text = Environment().from_string(LEGEND_TEMPLATE).render(
        {
            "B_fields": B_fields,
            "advanced_table": advanced_table,
            "data": dictionary_of_names_for_html,
            "download_mode": download_mode,
        }
    )
    for name in DEPRECATED:
        assert name not in text
    if download_mode:  # the README of the downloaded files documents every identifier and column
        for column in ["supercell_index", "atom_index_in_unit_cell", "cluster group", "label", "structure_id_pk"]:
            assert f'"{column}"' in text
        assert ("B_hf_norm" in text) is B_fields
        # the thresholds of the clustering are those of `aiida_muon.utils.clustering` (0.5 Angstrom, 0.05 eV)
        assert "0.5 Angstrom" in text and "0.05 eV" in text and "10^-3" not in text
    else:
        assert (dictionary_of_names_for_html["atom_index_in_unit_cell"] in text) is advanced_table


def test_readme_lists_the_files_that_are_written():
    """The files named in the README are those written by `produce_bitestream`."""
    text = Environment().from_string(LEGEND_TEMPLATE).render(
        {"B_fields": True, "advanced_table": False, "data": dictionary_of_names_for_html, "download_mode": True}
    )
    files_dict = {
        "table": make_table(),
        "table_all": make_table(),
        "structures": {
            "unit_cell": _FakeStructure(),
            "unit_cell_all": _FakeStructure(),
            "supercell_all": _FakeStructure(),
            "A": _FakeStructure(),
        },
        "distortions": {"A": pd.DataFrame({"x": [1]})},
        "readme": text,
    }
    payload = FindMuonModel.produce_bitestream(files_dict)
    with zipfile.ZipFile(io.BytesIO(base64.b64decode(payload))) as archive:
        written = {name.split("/")[-1] for name in archive.namelist()}
    for name in [
        "Summary_table.csv",
        "Summary_table_before_clustering.csv",
        "Allsites_unitcell.cif",
        "Allsites_before_clustering_unitcell.cif",
        "Allsites_before_clustering_supercell.cif",
        "Supercell_A.cif",
        "Distortion_supercell_A.csv",
    ]:
        assert name in written
        generic = name.replace("_A.", "_#.")
        assert generic in text, f"{generic} is not described in the README"


class _FakeStructure:
    def write(self, path, format=None):
        open(path, "w").write("fake cif")
