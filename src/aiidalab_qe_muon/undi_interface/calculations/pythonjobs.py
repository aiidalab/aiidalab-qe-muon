"""Functions executed through ``aiida-pythonjob`` and helpers to prepare their inputs.

``undi_run`` and ``compute_KT`` are plain Python functions: they are pickled by value
(``register_pickle_by_value=True``) and executed on the remote machine, where only ``undi``
needs to be installed. Hence, every import they need is done inside their body.

The ``prepare_*_inputs`` helpers return the inputs of an ``aiida_pythonjob.PythonJob``,
ready for ``WorkChain.submit(PythonJob, **inputs)``.
"""

from __future__ import annotations

import typing as t


def undi_run(
    structure,  # StructureData in AiiDA (provenance); the pythonjob deserializes it into ASE Atoms.
    B_mod=0.0,  # Units are Tesla.
    atom_as_muon="H",
    max_hdim=10e6,
    convergence_check=False,
    algorithm="fast",
    angular_integration_steps=7,
) -> dict:
    from undi.auto_analysis import execute_auto_analysis as execute_undi_analysis

    results = execute_undi_analysis(
        structure,
        B_mod=B_mod,
        atom_as_muon=atom_as_muon,
        max_hdim=max_hdim,
        convergence_check=convergence_check,
        algorithm=algorithm,
        angular_integration_steps=angular_integration_steps,
    )

    return {"results": results}


def compute_KT(
    structure,  # StructureData in AiiDA (provenance); the pythonjob deserializes it into ASE Atoms.
) -> dict:
    import numpy as np
    from undi.auto_analysis import compute_second_moments    
    from undi.kubo_toyabe.KT import kubo_toyabe

    t = np.linspace(0, 20e-6, 1000)  # time is seconds
    sm = compute_second_moments(structure)
    KT = kubo_toyabe(t, np.sum(list(sm.values())))

    return {
        "results": {
            "t": (np.array(t) * 1e6).tolist(),  # this time is in microseconds
            "KT": KT,
        },
    }


def _prepare(function, function_inputs, code, metadata):
    from aiida_pythonjob import prepare_pythonjob_inputs, spec

    return prepare_pythonjob_inputs(
        function=function,
        function_inputs=function_inputs,
        outputs_spec=spec.namespace(results=t.Any),
        register_pickle_by_value=True,
        code=code,
        metadata=dict(metadata or {}),
    )


def prepare_undi_run_inputs(
    structure,
    code,
    B_mod: float = 0.0,
    max_hdim: float = 1e1,
    atom_as_muon: str = "H",
    convergence_check: bool = False,
    algorithm: str = "fast",
    angular_integration_steps: int = 7,
    metadata: dict | None = None,
) -> dict:
    """Return the ``PythonJob`` inputs to run :func:`undi_run`.

    The scalar arguments are stored in ``function_inputs``, where the results viewer
    reads them back (``B_mod`` and ``max_hdim``).
    """
    return _prepare(
        undi_run,
        {
            "structure": structure,
            "B_mod": B_mod,
            "atom_as_muon": atom_as_muon,
            "max_hdim": max_hdim,
            "convergence_check": convergence_check,
            "algorithm": algorithm,
            "angular_integration_steps": angular_integration_steps,
        },
        code,
        metadata,
    )


def prepare_kt_inputs(
    structure,
    code,
    metadata: dict | None = None,
) -> dict:
    """Return the ``PythonJob`` inputs to run :func:`compute_KT`."""
    return _prepare(compute_KT, {"structure": structure}, code, metadata)
