"""Tests of the UNDI and Kubo-Toyabe shell jobs: the inputs, the parsers and real ``ShellJob`` runs on localhost.

The UNDI runs use a stand-in for ``python -m undi``, that has the command line of ``undi`` but does no physics (the real
UNDI needs its compiled extension). The Kubo-Toyabe job runs the real ``python -m undi.kubo_toyabe``, if ``undi`` has it.
"""

import importlib.util
import io
import json
import sys
import textwrap

import pytest
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import run_get_node

from aiidalab_qe_muon.undi_interface.calculations.shelljobs import (
    omp_threads,
    parse_kt_results,
    parse_undi_results,
    prepare_kt_inputs,
    prepare_undi_run_inputs,
    structure_to_file,
)

pytest.importorskip("aiida_shell")

UNDI_STAND_IN = textwrap.dedent(
    """
    import argparse, json, os, sys

    parser = argparse.ArgumentParser()
    parser.add_argument("structure")
    parser.add_argument("--Bmod", type=float, default=0.0)
    parser.add_argument("--max-hdim", type=int, default=1000)  # an integer, as in undi
    parser.add_argument("--atom-as-muon", default="H")
    parser.add_argument("--algorithm", default="fast")
    parser.add_argument("--angular-integration-steps", type=int, default=7)
    parser.add_argument("--convergence-check", action="store_true")
    parser.add_argument("--single-precision", action="store_true")
    args = parser.parse_args()

    print("progress bar and warnings go to stderr", file=sys.stderr)
    if args.Bmod < 0:
        sys.exit(3)

    n_atoms = int(open(args.structure).readline())
    results = [{
        "cluster_isotopes": {"Cu": 63}, "spins": {"Cu": 1.5}, "probability": 1.0,
        "B_ext": args.Bmod, "max_hdim": args.max_hdim, "convergence_check": args.convergence_check,
        "atom_as_muon": args.atom_as_muon, "algorithm": args.algorithm,
        "angular_integration_steps": args.angular_integration_steps, "n_atoms": n_atoms,
        "single_precision": args.single_precision, "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "t": [0.0, 1e-6], "signal_z_lf": [1.0, 0.5],
    }]
    json.dump(results, open("results.json", "w"))
    """
)

KT_STAND_IN = textwrap.dedent(
    """
    import json, sys
    n_atoms = int(open(sys.argv[1]).readline())
    json.dump({"t": [0.0, 20.0], "KT": [1.0, 0.33], "n_atoms": n_atoms}, open("KT.json", "w"))
    """
)


@pytest.fixture(scope="module")
def stand_in_undi(tmp_path_factory):
    """A directory with a fake ``undi`` package (first on the path of the jobs): ``python -m undi``, ``-m undi.kubo_toyabe``."""
    root = tmp_path_factory.mktemp("stand_in_undi")
    package = root / "undi"
    (package / "kubo_toyabe").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "kubo_toyabe" / "__init__.py").write_text("")
    (package / "__main__.py").write_text(UNDI_STAND_IN)
    (package / "kubo_toyabe" / "__main__.py").write_text(KT_STAND_IN)
    return root


@pytest.fixture
def make_code(aiida_localhost):
    """A `core.shell` python code on localhost, with the given directory first on its PYTHONPATH."""
    aiida_localhost.set_minimum_job_poll_interval(0.0)
    aiida_localhost.configure(safe_interval=0.0)

    def _make_code(label, pythonpath=None):
        return orm.InstalledCode(
            label=label,
            computer=aiida_localhost,
            filepath_executable=sys.executable,
            default_calc_job_plugin="core.shell",
            prepend_text=f"export PYTHONPATH={pythonpath}:$PYTHONPATH" if pythonpath else "",
        ).store()

    return _make_code


@pytest.fixture
def code(make_code, stand_in_undi):
    return make_code("python-stand-in-undi", stand_in_undi)


@pytest.fixture
def structure():
    """Cu cell and the muon (H), as the last site."""
    structure = orm.StructureData(cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    structure.append_atom(position=(1.5, 0.3, 0.2), symbols="H")
    return structure.store()


@pytest.fixture
def structure_file(structure):
    return structure_to_file(structure)


# ---- the structure file


def test_structure_file_keeps_cell_pbc_and_order(structure, structure_file):
    """The muon must stay the last atom; the file is read back as the same structure."""
    from ase.io import read

    assert structure_file.filename == "structure.xyz"
    atoms = read(io.StringIO(structure_file.get_content()), format="extxyz")
    expected = structure.get_ase()
    assert atoms.get_chemical_symbols() == expected.get_chemical_symbols() == ["Cu", "H"]
    assert atoms.positions == pytest.approx(expected.positions)
    assert atoms.cell[:] == pytest.approx(expected.cell[:])
    assert list(atoms.pbc) == list(expected.pbc)


def test_structure_file_is_a_calcfunction_of_the_structure(structure, structure_file):
    creator = structure_file.creator
    assert creator.process_label == "structure_to_file"
    assert [link.node.pk for link in creator.base.links.get_incoming(link_type=LinkType.INPUT_CALC).all()] == [
        structure.pk
    ]


# ---- the inputs of the jobs


def test_undi_inputs(code, structure_file):
    inputs = prepare_undi_run_inputs(
        structure_file, code, B_mod=0.002, max_hdim=1e4, atom_as_muon="H", convergence_check=False,
        algorithm="fast", angular_integration_steps=7, metadata={"options": {"max_wallclock_seconds": 60}},
    )
    assert inputs["arguments"] == [
        "-m", "undi", "{structure}", "--Bmod", "{B_mod}", "--max-hdim", "{max_hdim}", "--atom-as-muon",
        "{atom_as_muon}", "--algorithm", "{algorithm}", "--angular-integration-steps", "{angular_integration_steps}",
        "--single-precision",  # the default
    ]
    nodes = inputs["nodes"]
    assert set(nodes) == {
        "structure", "B_mod", "max_hdim", "atom_as_muon", "algorithm", "angular_integration_steps", "single_precision"
    }
    assert nodes["single_precision"].value is True
    assert nodes["structure"].pk == structure_file.pk
    assert (nodes["B_mod"].value, nodes["atom_as_muon"].value, nodes["algorithm"].value) == (0.002, "H", "fast")
    # the command line takes an integer: 1e4 must not become "10000.0"
    assert isinstance(nodes["max_hdim"], orm.Int) and str(nodes["max_hdim"].value) == "10000"
    # the file is for the parser only: not in `outputs`, as a missing one would hide why the command failed
    assert inputs["outputs"] is None
    assert inputs["metadata"]["options"]["additional_retrieve"] == ["results.json"]
    assert callable(inputs["parser"])


def test_convergence_check_is_a_bare_flag(code, structure_file):
    on = prepare_undi_run_inputs(structure_file, code, convergence_check=True)["arguments"]
    off = prepare_undi_run_inputs(structure_file, code, convergence_check=False)["arguments"]
    assert on[-1] == "--convergence-check" and on[:-1] == off
    assert "--convergence-check" not in off


def test_single_precision_is_the_default_and_can_be_turned_off(code, structure_file):
    single = prepare_undi_run_inputs(structure_file, code)
    double = prepare_undi_run_inputs(structure_file, code, single_precision=False)
    assert "--single-precision" in single["arguments"] and single["nodes"]["single_precision"].value is True
    assert "--single-precision" not in double["arguments"] and double["nodes"]["single_precision"].value is False
    # the flag is the only difference
    assert [a for a in single["arguments"] if a != "--single-precision"] == double["arguments"]


def test_stderr_is_redirected_unless_asked_otherwise(code, structure_file):
    """undi writes progress (tqdm) and warnings to stderr; a non-empty stderr would fail the job."""
    options = prepare_undi_run_inputs(structure_file, code, metadata={"options": {"max_wallclock_seconds": 60}})[
        "metadata"
    ]["options"]
    assert options == {
        "max_wallclock_seconds": 60,
        "redirect_stderr": True,
        "output_filename": "stdout",
        "prepend_text": "export OMP_NUM_THREADS=1",  # no resources: one CPU
        "additional_retrieve": ["results.json"],
    }
    options = prepare_kt_inputs(structure_file, code, metadata={"options": {"redirect_stderr": False}})["metadata"][
        "options"
    ]
    assert options["redirect_stderr"] is False
    assert prepare_kt_inputs(structure_file, code)["metadata"]["options"] == {
        "redirect_stderr": True,
        "output_filename": "stdout",
        "prepend_text": "export OMP_NUM_THREADS=1",
        "additional_retrieve": ["KT.json"],
    }


# ---- the OpenMP threads follow the CPUs of the job


@pytest.mark.parametrize(
    "resources, threads",
    [
        (None, 1),
        ({}, 1),
        ({"num_machines": 1}, 1),
        ({"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 6}, 6),  # `cpus per task` of the app
        ({"num_machines": 1, "num_mpiprocs_per_machine": 2, "num_cores_per_mpiproc": 4}, 8),
        ({"num_machines": 3, "num_mpiprocs_per_machine": 2}, 2),  # the command runs on one machine
        ({"num_cpus": 12}, 12),  # HyperQueue
    ],
)
def test_omp_threads_are_the_cpus_of_the_job(resources, threads):
    assert omp_threads(resources) == threads


def test_omp_threads_are_exported_from_the_resources(code, structure_file):
    options = prepare_undi_run_inputs(
        structure_file, code, metadata={"options": {"resources": {"num_cpus": 4}, "prepend_text": "module load x"}}
    )["metadata"]["options"]
    # the text of the user stays, the threads are added
    assert options["prepend_text"] == "module load x\nexport OMP_NUM_THREADS=4"
    assert prepare_kt_inputs(structure_file, code, metadata={"options": {"resources": {"num_cpus": 4}}})["metadata"][
        "options"
    ]["prepend_text"] == "export OMP_NUM_THREADS=4"


@pytest.mark.parametrize("key", ["prepend_text", "custom_scheduler_commands"])
def test_an_explicit_omp_num_threads_is_respected(code, structure_file, key):
    options = prepare_undi_run_inputs(
        structure_file, code, metadata={"options": {"resources": {"num_cpus": 4}, key: "export OMP_NUM_THREADS=2"}}
    )["metadata"]["options"]
    assert options[key] == "export OMP_NUM_THREADS=2"
    assert "OMP_NUM_THREADS" not in options.get("prepend_text", "").replace("export OMP_NUM_THREADS=2", "")


def test_kt_inputs(code, structure_file):
    inputs = prepare_kt_inputs(structure_file, code)
    assert inputs["arguments"] == ["-m", "undi.kubo_toyabe", "{structure}"]
    assert set(inputs["nodes"]) == {"structure"}
    assert inputs["metadata"]["options"]["additional_retrieve"] == ["KT.json"]


# ---- the parsers


def test_parsers_defer_to_the_exit_status_of_a_failed_command(tmp_path):
    (tmp_path / "status").write_text("3")
    assert parse_undi_results(tmp_path) == {} and parse_kt_results(tmp_path) == {}


def test_parsers_complain_if_a_successful_command_wrote_nothing(tmp_path):
    (tmp_path / "status").write_text("0")
    with pytest.raises(FileNotFoundError, match="results.json"):
        parse_undi_results(tmp_path)
    with pytest.raises(FileNotFoundError, match="KT.json"):
        parse_kt_results(tmp_path)


def test_parsers(tmp_path):
    (tmp_path / "results.json").write_text(json.dumps([{"B_ext": 0.0, "t": [0.0, 1.0]}]))
    (tmp_path / "KT.json").write_text(json.dumps({"t": [0.0], "KT": [1.0]}))
    results = parse_undi_results(tmp_path)["results"]
    assert isinstance(results, orm.List) and results.get_list() == [{"B_ext": 0.0, "t": [0.0, 1.0]}]
    kt = parse_kt_results(tmp_path)["results"]
    assert isinstance(kt, orm.Dict) and kt.get_dict() == {"t": [0.0], "KT": [1.0]}


# ---- real jobs


def run_job(inputs):
    from aiida.plugins import CalculationFactory

    return run_get_node(CalculationFactory("core.shell"), **inputs)


def test_undi_job_runs_the_command_line(code, structure_file):
    """The placeholders become the arguments of `python -m undi`; its stderr does not fail the job."""
    results, node = run_job(
        prepare_undi_run_inputs(
            structure_file, code, B_mod=0.002, max_hdim=1e4, convergence_check=True, algorithm="fast",
            angular_integration_steps=5,
        )
    )
    assert node.is_finished_ok, node.exit_message
    (cluster,) = results["results"].get_list()
    assert (cluster["B_ext"], cluster["max_hdim"], cluster["convergence_check"]) == (0.002, 10000, True)
    assert (cluster["atom_as_muon"], cluster["algorithm"], cluster["angular_integration_steps"]) == ("H", "fast", 5)
    assert cluster["n_atoms"] == 2  # the structure file, with the muon
    assert cluster["single_precision"] is True and cluster["omp_threads"] == "1"
    # what the results viewer reads
    assert node.inputs.nodes.B_mod.value == 0.002 and node.inputs.nodes.max_hdim.value == 10000


def test_job_runs_with_the_threads_and_the_precision_asked(code, structure_file):
    """`OMP_NUM_THREADS` is the CPUs of the job, in the environment of the command; `--single-precision` can be off."""
    results, node = run_job(
        prepare_undi_run_inputs(
            structure_file, code, single_precision=False,
            metadata={"options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 3}}},
        )
    )
    assert node.is_finished_ok, node.exit_message
    (cluster,) = results["results"].get_list()
    assert cluster["omp_threads"] == "3" and cluster["single_precision"] is False
    assert node.inputs.nodes.single_precision.value is False


_submit_dirs = iter(range(1000))


def _submit_script(inputs, tmp_path, monkeypatch):
    """Dry run: the `_aiidasubmit.sh` that would be submitted to the scheduler (nothing is run nor stored)."""
    from aiida.engine import run
    from aiida.plugins import CalculationFactory

    workdir = tmp_path / f"dry_run_{next(_submit_dirs)}"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    inputs = dict(inputs)
    inputs["metadata"] = {**inputs["metadata"], "dry_run": True, "store_provenance": False}
    run(CalculationFactory("core.shell"), **inputs)
    (script,) = (workdir / "submit_test").glob("*/_aiidasubmit.sh")
    return script.read_text()


def _exported_threads(script):
    """The values of `OMP_NUM_THREADS` exported by the script, in order (the last one is the one that counts)."""
    import re

    return [int(value) for value in re.findall(r"export OMP_NUM_THREADS=(\d+)", script)]


@pytest.mark.parametrize(
    ("resources", "threads"),
    [
        # what the app builds (`create_resource_config`): nodes, ntasks per node, cpus per task
        ({"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 6}, 6),
        ({"num_machines": 1, "num_mpiprocs_per_machine": 2, "num_cores_per_mpiproc": 4}, 8),
        ({"num_machines": 1, "num_mpiprocs_per_machine": 1}, 1),
    ],
)
def test_the_submit_script_exports_the_threads_of_the_resources(
    code, structure_file, tmp_path, monkeypatch, resources, threads
):
    """The export is really in the script of the job, before the command, for the UNDI and the Kubo-Toyabe jobs.

    The direct scheduler also exports the cores per process in the header of the script: ours comes after it, so the
    last export (the effective one) is the CPUs of the job (processes x cores).
    """
    for inputs in (
        prepare_undi_run_inputs(structure_file, code, metadata={"options": {"resources": resources}}),
        prepare_kt_inputs(structure_file, code, metadata={"options": {"resources": resources}}),
    ):
        script = _submit_script(inputs, tmp_path, monkeypatch)
        assert _exported_threads(script)[-1] == threads, script
        assert script.rindex("export OMP_NUM_THREADS") < script.index("'-m'"), script  # set before the command runs


def test_the_submit_script_ends_with_the_explicit_value(code, structure_file, tmp_path, monkeypatch):
    """`OMP_NUM_THREADS` in the options is the effective value (and we do not add another one after it)."""
    inputs = prepare_undi_run_inputs(
        structure_file, code,
        metadata={"options": {"resources": {"num_machines": 1, "num_mpiprocs_per_machine": 1, "num_cores_per_mpiproc": 6},
                              "prepend_text": "export OMP_NUM_THREADS=3"}},
    )
    script = _submit_script(inputs, tmp_path, monkeypatch)
    assert _exported_threads(script)[-1] == 3, script


def test_job_has_the_output_filename_the_app_reads(code, structure_file):
    """The app shows the log of a job from the `output_filename` attribute (it fails if there is none)."""
    results, node = run_job(prepare_undi_run_inputs(structure_file, code))
    assert node.is_finished_ok, node.exit_message
    assert node.base.attributes.get("output_filename") == "stdout"  # what the app does, without a default
    # the output is the same file as before, with stderr in it
    assert "progress bar and warnings go to stderr" in results["stdout"].get_content()


def test_failed_command_fails_the_job(code, structure_file):
    _, node = run_job(prepare_undi_run_inputs(structure_file, code, B_mod=-1.0))
    # the reason is the exit status of the command, not a parser that did not find the file
    assert node.is_failed and "parser" not in node.exit_message and "3" in node.exit_message


def test_kubo_toyabe_job_stand_in(code, structure_file):
    results, node = run_job(prepare_kt_inputs(structure_file, code))
    assert node.is_finished_ok, node.exit_message
    assert results["results"].get_dict()["KT"] == [1.0, 0.33]


@pytest.mark.skipif(
    importlib.util.find_spec("undi") is None or importlib.util.find_spec("ase") is None,
    reason="undi (with ase) is not installed",
)
def test_kubo_toyabe_job_with_the_real_undi(make_code, structure_file):
    """`python -m undi.kubo_toyabe` of the installed undi: t in microseconds, full polarization at t=0."""
    import undi

    pytest.importorskip("undi.kubo_toyabe.__main__", reason="this undi has no Kubo-Toyabe command line")
    root = importlib.util.find_spec("undi").submodule_search_locations[0]
    code = make_code("python-real-undi", pythonpath=str(__import__("pathlib").Path(root).parent))
    results, node = run_job(prepare_kt_inputs(structure_file, code))
    assert node.is_finished_ok, node.exit_message
    kt = results["results"].get_dict()
    assert len(kt["t"]) == len(kt["KT"]) == 1000
    assert kt["t"][-1] == pytest.approx(20.0) and kt["KT"][0] == pytest.approx(1.0)
