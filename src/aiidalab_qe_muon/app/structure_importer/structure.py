"""Module to provide functionality to import structures."""

import datetime
import functools
import html
import io
import pathlib
import tempfile
from collections import OrderedDict

import ase
import ipywidgets as ipw
import numpy as np
import spglib
import traitlets as tl
from aiida import engine, orm, plugins

from aiidalab_widgets_base.data import LigandSelectorWidget
from aiidalab_widgets_base.utils import StatusHTML, get_ase_from_file, get_formula
from aiidalab_widgets_base.viewers import StructureDataViewer

from pymatgen.io.cif import CifParser
from aiidalab_widgets_base import (
    StructureUploadWidget,
)
from pymatgen.core import Structure

def get_pymatgen_from_file(fname, file_format="mcif", primitive=True):  # pylint: disable=redefined-builtin
    """Get the first pymatgen structure of a (m)cif file.

    :raises ValueError: if the file cannot be read, with the reason (and the original exception as its cause).
    """
    name = pathlib.Path(fname).name
    try:
        parser = CifParser(fname)
        # `get_structures` was renamed `parse_structures` in recent pymatgen versions.
        read = getattr(parser, "parse_structures", None) or parser.get_structures
        structures = read(primitive=primitive)
    except Exception as exc:
        raise ValueError(
            f"Could not read the file {name}: {type(exc).__name__}: {exc}"
        ) from exc
    if not structures:
        raise ValueError(f"No structure found in the file {name}")
    return structures[0]


class ImportMagnetism(StructureUploadWidget):
    """
    Import a structure by reading mcif file, for magnetism.
    Up to now, this is meant to work with muons.
    """

    structure = tl.Union(
        [tl.Instance(Structure), tl.Instance(orm.Data)], allow_none=True
    )

    def __init__(
        self, title="Upload mcif", description="Upload Structure", allow_trajectories=False
    ):
        super().__init__()
        self.title = title
        self.message = ipw.HTML(
            "<b>Please note:</b> the magnetic moments of the uploaded mcif are kept with the structure and "
            "are used in the muon simulations, after being made collinear "
            "(the <i>Muon settings</i> panel of step 2 shows both the mcif and the collinear moments).<br>"
            "<b>They will be overridden</b> if you set the magnetism in step 2, i.e. the "
            "<i>Initial magnetic moments</i> in the <i>Magnetism</i> tab of the <i>Advanced settings</i>."
        )
        self.file_upload_subwidget1 = ipw.Checkbox(
            description="Use conventional (select before the upload)",
            indent=False,
            value=False,
        )
        self.file_upload = ipw.FileUpload(
            description=description, multiple=False, layout={"width": "initial"}
        )
        
        # Whether to allow uploading multiple structures from a single file.
        # In this case, we create TrajectoryData node.
        supported_formats = ipw.HTML(
            """<a Supported format is mcif. </a>"""
        )
        self._status_message = StatusHTML(clear_after=5)
        self.file_upload.observe(self._on_file_upload, names="value")
        
        # why a file could not be read (shown until the next upload)
        self.error_message = ipw.HTML()
        self.children = tuple(
            [self.file_upload_subwidget1, self.message, self.error_message]
        ) + self.children
        
    def _validate_and_fix_pymatgen_cell(self, structure):
    
        structure_aiida = orm.StructureData(pymatgen=structure)
        if "magmom" in structure.site_properties.keys():
            magmoms = structure.site_properties["magmom"]
            magmom = orm.List([list(magmom) for magmom in magmoms])  
            structure_aiida.base.extras.set("magmom",magmom)
        
        return structure_aiida
    
    def _read_structure(self, fname, content):
        suffix = "".join(pathlib.Path(fname).suffixes)            

        with tempfile.NamedTemporaryFile(suffix=suffix) as temp_file:
            temp_file.write(content)
            temp_file.flush()
            try:
                structures = get_pymatgen_from_file(
                    temp_file.name, primitive=not self.file_upload_subwidget1.value
                )
            except ValueError as exc:
                # the reason is in the message of the exception, which also chains the original one
                reason = html.escape(str(exc).replace(pathlib.Path(temp_file.name).name, fname))
                self.error_message.value = (
                    f'<div class="alert alert-danger"><b>ERROR:</b> <code>{reason}</code></div>'
                )
                return None
            self.error_message.value = ""

            return self._validate_and_fix_pymatgen_cell(structures)
        