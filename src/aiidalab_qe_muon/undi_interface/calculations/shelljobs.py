"""UNDI and Kubo-Toyabe as shell jobs (``aiida-shell``): the command line of ``undi`` and its parsers.

The jobs run the command line interfaces of ``undi``, on the compute resource, with the python of the ``undi`` code
(which needs ``undi[undi]`` installed):

* ``python -m undi structure.xyz --Bmod ... --max-hdim ...``: one UNDI run, results in ``results.json``;
* ``python -m undi.kubo_toyabe structure.xyz``: the Kubo-Toyabe polarization, in ``KT.json``.

The ``prepare_*_inputs`` functions return the inputs of an ``aiida_shell.ShellJob``, ready for
``WorkChain.submit(ShellJob, **inputs)``. The scalars are input nodes (``B_mod``, ``max_hdim``, ...): the results viewer
reads them back from ``job.inputs.nodes``.
"""

from __future__ import annotations

import io
import json
import pathlib

from aiida import orm
from aiida.engine import calcfunction

STRUCTURE_FILENAME = "structure.xyz"
UNDI_RESULTS_FILENAME = "results.json"
KT_RESULTS_FILENAME = "KT.json"


@calcfunction
def structure_to_file(structure: orm.StructureData) -> orm.SinglefileData:
    """The structure as a file the ``undi`` command line can read (extended xyz).

    Extended xyz keeps the cell, the periodic boundary conditions and the order of the atoms, which matters: the muon
    has to be the last one.
    """
    from ase.io import write

    buffer = io.StringIO()
    write(buffer, structure.get_ase(), format="extxyz")
    return orm.SinglefileData(
        io.BytesIO(buffer.getvalue().encode()), filename=STRUCTURE_FILENAME
    )


def _read_json(dirpath: pathlib.Path, filename: str):
    """The content of ``filename`` in ``dirpath``, or ``None`` if the command failed (so did not write it).

    If the command failed, the parser returns nothing, and the job fails with the exit status of the command.
    If it succeeded but wrote no file, that is an error of the parser.
    """
    dirpath = pathlib.Path(dirpath)
    filepath = dirpath / filename
    if not filepath.exists():
        try:
            failed = int((dirpath / "status").read_text()) != 0
        except (FileNotFoundError, ValueError):
            failed = False
        if failed:
            return None
        raise FileNotFoundError(f"The command did not write {filename}")
    with filepath.open() as handle:
        return json.load(handle)


def parse_undi_results(dirpath: pathlib.Path) -> dict:
    """Parser of the UNDI job: ``results.json`` as the ``results`` list (one dictionary per cluster of isotopes)."""
    results = _read_json(dirpath, UNDI_RESULTS_FILENAME)
    return {} if results is None else {"results": orm.List(results)}


def parse_kt_results(dirpath: pathlib.Path) -> dict:
    """Parser of the Kubo-Toyabe job: ``KT.json`` as the ``results`` dictionary, with ``t`` (µs) and ``KT``."""
    results = _read_json(dirpath, KT_RESULTS_FILENAME)
    return {} if results is None else {"results": orm.Dict(results)}


def omp_threads(resources: dict | None) -> int:
    """The number of OpenMP threads of a job: the CPUs it asks for.

    The C++ part of ``undi`` is parallel with OpenMP, and the job runs the command once (it is not MPI), so the threads
    are all the CPUs of the job: ``num_cpus`` (HyperQueue), or the MPI processes per machine times the cores per
    process (the "ntasks" and "cpus per task" of the app). One thread if the resources do not tell.
    """
    resources = resources or {}
    if "num_cpus" in resources:
        return max(1, int(resources["num_cpus"]))
    return max(1, int(resources.get("num_mpiprocs_per_machine", 1)) * int(resources.get("num_cores_per_mpiproc", 1)))


def _sets_omp_threads(options: dict) -> bool:
    return any("OMP_NUM_THREADS" in str(options.get(key) or "") for key in ("prepend_text", "custom_scheduler_commands"))


def _prepare(code, arguments, nodes, results_filename, parser, metadata):
    from aiida_shell import ShellJob
    from aiida_shell.launch import prepare_shell_job_inputs

    metadata = dict(metadata or {})
    options = dict(metadata.get("options") or {})
    # `undi` writes progress (tqdm) and warnings to stderr: with the default, any of them would fail the job.
    # The exit status of the command is what tells if it failed.
    options.setdefault("redirect_stderr", True)
    # `aiida-shell` writes the output to `stdout` by default, without setting the `output_filename` option. The app
    # reads that attribute of the job to show its log, and fails ("attribute `output_filename` does not exist") if it
    # is missing. Setting it to the same file changes nothing else.
    options.setdefault("output_filename", ShellJob.FILENAME_STDOUT)
    # The OpenMP threads follow the CPUs of the job (all the cores of the machine, if not set, which can be more than
    # the job has). An explicit `OMP_NUM_THREADS` in the options is respected.
    if not _sets_omp_threads(options):
        export = f"export OMP_NUM_THREADS={omp_threads(options.get('resources'))}"
        options["prepend_text"] = "\n".join(filter(None, [options.get("prepend_text"), export]))
    # The file is only for the parser (it is not an `outputs` of the job: a missing one would hide the exit status of
    # the command, when it fails).
    options["additional_retrieve"] = [*options.get("additional_retrieve", []), results_filename]
    metadata["options"] = options

    return prepare_shell_job_inputs(
        command=code,
        arguments=arguments,
        nodes=nodes,
        parser=parser,
        metadata=metadata,
    )


def prepare_undi_run_inputs(
    structure_file,
    code,
    B_mod: float = 0.0,
    max_hdim: float = 1e1,
    atom_as_muon: str = "H",
    convergence_check: bool = False,
    algorithm: str = "fast",
    angular_integration_steps: int = 7,
    single_precision: bool = True,
    metadata: dict | None = None,
) -> dict:
    """Return the ``ShellJob`` inputs of one UNDI run: ``python -m undi``.

    :param structure_file: the ``SinglefileData`` with the structure (see :func:`structure_to_file`).
    :param code: the code of the python with ``undi``.
    :param single_precision: run the C++ algorithm in single precision (``--single-precision``): about half the memory
        and faster, with a small loss of accuracy.
    """
    arguments = [
        "-m",
        "undi",
        "{structure}",
        "--Bmod",
        "{B_mod}",
        "--max-hdim",
        "{max_hdim}",
        "--atom-as-muon",
        "{atom_as_muon}",
        "--algorithm",
        "{algorithm}",
        "--angular-integration-steps",
        "{angular_integration_steps}",
    ]
    if single_precision:
        arguments.append("--single-precision")
    if convergence_check:
        arguments.append("--convergence-check")

    nodes = {
        "structure": structure_file,
        "B_mod": orm.Float(B_mod),
        "max_hdim": orm.Int(int(max_hdim)),  # the command line takes an integer
        "atom_as_muon": orm.Str(atom_as_muon),
        "algorithm": orm.Str(algorithm),
        "angular_integration_steps": orm.Int(angular_integration_steps),
        # not an argument with a value (the flag is above): an input, to keep it in the provenance
        "single_precision": orm.Bool(single_precision),
    }
    return _prepare(code, arguments, nodes, UNDI_RESULTS_FILENAME, parse_undi_results, metadata)


def prepare_kt_inputs(structure_file, code, metadata: dict | None = None) -> dict:
    """Return the ``ShellJob`` inputs of the Kubo-Toyabe polarization: ``python -m undi.kubo_toyabe``."""
    return _prepare(
        code,
        ["-m", "undi.kubo_toyabe", "{structure}"],
        {"structure": structure_file},
        KT_RESULTS_FILENAME,
        parse_kt_results,
        metadata,
    )
