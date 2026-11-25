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
