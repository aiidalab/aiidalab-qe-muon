"""Magnetism helpers for the muon app.

Kept free of any widget import, so that it can be used (and tested) without the AiiDAlab QE app.
"""

from __future__ import annotations

import numpy as np


def collinearize_magmoms(structure, magmoms, half: bool = True) -> dict:
    """Make the magnetic moments of a structure collinear, as the muon workflow does, without storing anything.

    This runs the plain function behind the ``make_collinear_getmag_kind`` calcfunction of ``aiida-muon``
    (``get_collinear_mag_kindname``), so the result is the one the workflow will use, but no provenance is created.

    :param structure: the ``StructureData`` (it can be unstored).
    :param magmoms: one ``[mx, my, mz]`` per site, in the order of the sites of the structure.
    :param half: as in ``aiida-muon``: if ``True`` the starting magnetization of the magnetic kinds is +-0.5.
    :return: a dictionary with ``rows`` (one dictionary per site: ``site``, ``element``, ``kind``, ``magmom``, the
        collinear ``[0, 0, mz]`` moment, and ``start_mag``, the starting magnetization of its kind, ``0.0`` if the
        site is not magnetic) and ``start_mag``, the starting magnetization of every magnetic kind.
    :raises ValueError: if the number of moments does not match the number of sites.
    """
    from aiida_muon.utils.magnetism import get_collinear_mag_kindname
    from pymatgen.electronic_structure.core import Magmom

    if len(magmoms) != len(structure.sites):
        raise ValueError(
            f"{len(magmoms)} magnetic moments for {len(structure.sites)} sites"
        )

    # a fresh pymatgen structure: the function sets the site properties in place.
    p_st = structure.get_pymatgen_structure()
    collinear_st, start_mag = get_collinear_mag_kindname(
        p_st, [Magmom(m) for m in magmoms], half
    )

    rows = []
    for index, site in enumerate(collinear_st):
        kind = site.properties.get("kind_name")
        moment = np.asarray(Magmom(site.properties["magmom"]).moment, dtype=float)
        rows.append(
            {
                "site": index,
                "element": site.specie.symbol,
                "kind": kind or site.specie.symbol,
                "magmom": [float(x) for x in moment],
                "start_mag": float(start_mag.get(kind, 0.0)) if kind else 0.0,
            }
        )
    return {"rows": rows, "start_mag": dict(start_mag)}


def magmoms_table_html(kinds, magmoms, collinear_rows=(), error: str = "") -> str:
    """The table of the magnetic moments of the structure and their collinearization, as HTML.

    One row per site: the kind and the moment of the structure (e.g. from an mcif), then, as computed by
    :func:`collinearize_magmoms`, the new kind, the collinear moment along z and the starting magnetization
    given to Quantum ESPRESSO (in bold). The collinear columns show ``–`` if ``collinear_rows`` is empty or
    shorter than the moments; ``error`` is then shown below the table, if given.

    :param kinds: the kind name of each site of the structure.
    :param magmoms: one ``[mx, my, mz]`` per site.
    :param collinear_rows: the ``rows`` returned by :func:`collinearize_magmoms`.
    :param error: why the collinear moments are not available, if so.
    """
    from html import escape

    cell = "<td style='padding:1px 8px;'>{}</td>"
    head = "<th style='padding:1px 8px;'>{}</th>"
    group = "<th colspan='{}' style='padding:1px 8px;border-bottom:1px solid grey;'>{}</th>"

    header_groups = group.format(5, "From the structure (e.g. mcif)") + group.format(
        3, "Collinearized (used by the workflow)"
    )
    header = "".join(
        head.format(title)
        for title in [
            "Site",
            "Kind",
            "mx (μ<sub>B</sub>)",
            "my",
            "mz",
            "New kind",
            "mz (μ<sub>B</sub>)",
            "Starting magnetization",
        ]
    )

    rows = []
    for index, magmom in enumerate(magmoms):
        kind = kinds[index] if index < len(kinds) else ""
        values = [index, escape(str(kind))] + [f"{x:.3f}" for x in magmom]
        if index < len(collinear_rows):
            row = collinear_rows[index]
            values += [
                escape(str(row["kind"])),
                f"{row['magmom'][2]:.3f}",
                f"<b>{row['start_mag']:+.1f}</b>",
            ]
        else:
            values += ["–", "–", "–"]
        rows.append("<tr>" + "".join(cell.format(v) for v in values) + "</tr>")

    note = (
        "<span style='color:grey;font-size:0.85em;'>Collinearization as done by <code>aiida-muon</code> for the "
        "workflow: the moments are made collinear along z and the kinds are split by element and spin. "
        "The starting magnetization is the value given to Quantum ESPRESSO.</span>"
    )
    if error:
        note += (
            "<br><span style='color:#b00;font-size:0.85em;'>The collinear moments could not be computed: "
            f"{escape(error)}</span>"
        )
    return (
        f"<table style='font-size:0.85em;border-collapse:collapse;'><tr>{header_groups}</tr><tr>{header}</tr>"
        f"{''.join(rows)}</table>{note}"
    )
