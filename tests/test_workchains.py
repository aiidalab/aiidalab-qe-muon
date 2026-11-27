"""Tests of the UNDI / Kubo-Toyabe work chains.

``undi`` is not needed: the ``ShellJob`` is replaced by a fake work chain that returns the same outputs, so that the
provenance (call link labels, inputs, outputs) can be checked. The real ``ShellJob`` is in ``test_shelljobs.py``.
"""

import pytest
from _stubs import fake_modules
from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import WorkChain, run_get_node

from aiidalab_qe_muon.undi_interface.calculations.shelljobs import structure_to_file
from aiidalab_qe_muon.undi_interface.workflows import workchains
from aiidalab_qe_muon.undi_interface.workflows.workchains import (
    MultiSitesWorkChain,
    UndiAndKuboToyabeWorkChain,
    UndiMultiAnalysisWorkChain,
)


class FakeShellJob(WorkChain):
    """Stand-in for ``aiida_shell.ShellJob``: ``results`` is a ``List`` for UNDI and a ``Dict`` for KT.

    The outputs are stored by hand, as a work chain cannot return unstored data (a real ``ShellJob`` is a calcjob).
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace("nodes", dynamic=True)
        spec.input("code", valid_type=orm.AbstractCode, required=False)
        spec.input("fail", valid_type=orm.Bool, required=False)
        spec.output("results", valid_type=orm.Data)
        spec.exit_code(300, "ERROR_FAKE", message="Fake failure.")
        spec.outline(cls.run_fake)

    def run_fake(self):
        inputs = self.inputs.nodes
        if "fail" in self.inputs and self.inputs.fail.value:
            return self.exit_codes.ERROR_FAKE
        if "B_mod" in inputs:
            self.out("results", orm.List([{"B_mod": inputs.B_mod.value, "max_hdim": inputs.max_hdim.value}]).store())
        else:
            self.out("results", orm.Dict({"t": [0.0], "KT": [1.0]}).store())


@pytest.fixture
def fake_shelljob(monkeypatch):
    """Replace the ``ShellJob`` and the preparation of its inputs by the fake."""
    state = {"fail_B_mod": None}

    def prepare_undi_run_inputs(structure_file, code, B_mod, max_hdim, metadata=None, **kwargs):
        fail = state["fail_B_mod"] is not None and B_mod == state["fail_B_mod"]
        inputs = {
            "nodes": {"structure": structure_file, "B_mod": orm.Float(B_mod), "max_hdim": orm.Int(int(max_hdim))},
            "code": code,
        }
        if fail:
            inputs["fail"] = orm.Bool(True)
        return inputs

    def prepare_kt_inputs(structure_file, code, metadata=None):
        return {"nodes": {"structure": structure_file}, "code": code}

    monkeypatch.setattr(workchains, "_get_shelljob_class", lambda: FakeShellJob)
    monkeypatch.setattr(workchains, "prepare_undi_run_inputs", prepare_undi_run_inputs)
    monkeypatch.setattr(workchains, "prepare_kt_inputs", prepare_kt_inputs)
    return state


@pytest.fixture
def code(aiida_localhost):
    return orm.InstalledCode(
        label="fake-shelljob",
        computer=aiida_localhost,
        filepath_executable="/bin/true",
    ).store()


@pytest.fixture
def structure():
    """Cu and the muon (H), as the last site."""
    structure = orm.StructureData(cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    structure.append_atom(position=(1.5, 0, 0), symbols="H")
    return structure.store()


@pytest.fixture
def structure_file(structure):
    return structure_to_file(structure)


def labels_of(node, link_type=LinkType.CALL_WORK):
    return sorted(link.link_label for link in node.base.links.get_outgoing(link_type=link_type).all())


def test_specs():
    """The ports and exit codes the rest of the app relies on."""
    undi = UndiMultiAnalysisWorkChain.spec()
    assert {"structure_file", "code", "B_mods", "max_hdims", "convergence_check", "task_metadata"} <= set(undi.inputs)
    assert "results" in undi.outputs

    # only the chains that run the production have `max_hdim`, the other one runs what it is given
    assert "max_hdim" not in undi.inputs

    kt = UndiAndKuboToyabeWorkChain.spec()
    assert set(kt.outputs["results"]) == {"KT_task", "undi_conv_task", "undi_task"}
    assert "max_hdim" in kt.inputs

    sites = MultiSitesWorkChain.spec()
    assert sites.inputs["structure_group"].dynamic
    assert "max_hdim" in sites.inputs
    assert "results" in sites.outputs


def test_multiple_undi_analysis(fake_shelljob, code, structure_file):
    """B_mods x max_hdims runs, labelled iter_{i} with i running over the fields first."""
    results, node = run_get_node(
        UndiMultiAnalysisWorkChain,
        structure_file=structure_file,
        code=code,
        B_mods=orm.List([0.0, 1e-3]),
        max_hdims=orm.List([4, 16]),
    )
    assert node.is_finished_ok
    assert labels_of(node) == ["iter_0", "iter_1", "iter_2", "iter_3"]
    assert sorted(results["results"]) == ["iter_0", "iter_1", "iter_2", "iter_3"]
    called = {link.link_label: link.node for link in node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()}
    assert [
        (called[f"iter_{i}"].inputs.nodes.B_mod.value, called[f"iter_{i}"].inputs.nodes.max_hdim.value)
        for i in range(4)
    ] == [(0.0, 4.0), (0.0, 16.0), (1e-3, 4.0), (1e-3, 16.0)]


def test_undi_and_kubo_toyabe(fake_shelljob, code, structure):
    results, node = run_get_node(
        UndiAndKuboToyabeWorkChain,
        structure=structure,
        code=code,
        B_mods=orm.List([0.0, 1e-3]),
        max_hdims=orm.List([4, 16, 64]),
        max_hdim=orm.Int(16),
        convergence_check=orm.Bool(True),
    )
    assert node.is_finished_ok
    assert labels_of(node) == ["KuboToyabe_run", "convergence_check", "undi_runs"]
    assert isinstance(results["results"]["KT_task"], orm.Dict)
    # convergence check: B=0, every max_hdim. undi_runs: every field, the production max_hdim only.
    assert sorted(results["results"]["undi_conv_task"]) == ["iter_0", "iter_1", "iter_2"]
    assert sorted(results["results"]["undi_task"]) == ["iter_0", "iter_1"]
    assert [r["max_hdim"] for r in results["results"]["undi_task"]["iter_0"].get_list()] == [16.0]


def test_default_sizes(fake_shelljob, code, structure):
    """Convergence check with 10^2, 10^4 and 10^6, production runs with 10^6 (not tied to the list)."""
    results, node = run_get_node(
        UndiAndKuboToyabeWorkChain,
        structure=structure,
        code=code,
        B_mods=orm.List([0.0, 1e-3]),
        convergence_check=orm.Bool(True),
    )
    assert node.is_finished_ok
    jobs = {
        link.link_label: {job.link_label: job.node for job in link.node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()}
        for link in node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()
        if link.link_label in ("convergence_check", "undi_runs")
    }
    sizes = lambda label: [  # noqa: E731
        job.inputs.nodes.max_hdim.value for _, job in sorted(jobs[label].items())
    ]
    assert sizes("convergence_check") == [10**2, 10**4, 10**6]
    assert sizes("undi_runs") == [10**6, 10**6]  # one per field

    # the production size is its own input: it does not have to be in the list
    results, node = run_get_node(
        UndiAndKuboToyabeWorkChain,
        structure=structure,
        code=code,
        max_hdims=orm.List([4, 16]),
        max_hdim=orm.Int(32),
    )
    assert [r["max_hdim"] for r in results["results"]["undi_task"]["iter_0"].get_list()] == [32.0]


def test_undi_and_kubo_toyabe_without_convergence_check(fake_shelljob, code, structure):
    results, node = run_get_node(
        UndiAndKuboToyabeWorkChain,
        structure=structure,
        code=code,
        max_hdims=orm.List([4, 16]),
    )
    assert node.is_finished_ok
    assert labels_of(node) == ["KuboToyabe_run", "undi_runs"]
    assert "undi_conv_task" not in results["results"]


def test_multisites(fake_shelljob, code, structure):
    """One child per site, and the convergence check on the first one only."""
    second = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    second.append_atom(position=(0, 0, 0), symbols="Cu")
    second.append_atom(position=(2.0, 0, 0), symbols="H")
    second.store()
    results, node = run_get_node(
        MultiSitesWorkChain,
        structure_group={"3": structure, "7": second},
        code=code,
        B_mods=orm.List([0.0]),
        max_hdims=orm.List([4, 16]),
    )
    assert node.is_finished_ok
    assert labels_of(node) == ["polarization_structure_3", "polarization_structure_7"]
    assert sorted(results["results"]) == ["site_3", "site_7"]
    assert set(results["results"]["site_3"]) == {"KT_task", "undi_conv_task", "undi_task"}
    assert set(results["results"]["site_7"]) == {"KT_task", "undi_task"}
    # how `ImplantMuonWorkChain` picks the KT of the first site
    assert node.outputs.results["site_3"]["KT_task"].pk == results["results"]["site_3"]["KT_task"].pk

    # what the results viewer walks through: KT output -> KT job -> site work chain -> multisites -> all the sites.
    # With a real `PythonJob` (a calcjob) the KT job is `kt.creator`; the fake is a work chain, so it only returns it.
    kt = results["results"]["site_7"]["KT_task"]
    kt_job = next(
        link.node
        for link in kt.base.links.get_incoming(link_type=LinkType.RETURN).all()
        if link.node.process_label == "FakeShellJob"
    )
    assert kt_job.caller.process_label == "UndiAndKuboToyabeWorkChain"
    multisites = kt_job.caller.caller
    assert multisites.pk == node.pk
    assert sorted(called.pk for called in multisites.called) == sorted(
        link.node.pk for link in node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()
    )


def test_failed_child_fails_everything(fake_shelljob, code, structure):
    fake_shelljob["fail_B_mod"] = 1e-3
    _, node = run_get_node(
        MultiSitesWorkChain,
        structure_group={"0": structure},
        code=code,
        B_mods=orm.List([0.0, 1e-3]),
        max_hdims=orm.List([4, 16]),
    )
    assert node.is_failed
    assert node.exit_status == MultiSitesWorkChain.exit_codes.ERROR_SITE_FAILED.status
    site = node.called[0]
    assert site.exit_status == UndiAndKuboToyabeWorkChain.exit_codes.ERROR_UNDI_FAILED.status
    assert "results" not in node.outputs


def test_implant_muon_polarization_only(fake_shelljob, code, monkeypatch):
    """`ImplantMuonWorkChain` without the muon search: it submits the `MultiSitesWorkChain` and returns the first KT."""
    fake_modules(monkeypatch, "pylocfield", "pylocfield.ewald", "pylocfield.realspace")  # imported by `aiida-muon`, not used here
    module = pytest.importorskip("aiidalab_qe_muon.workflows.implantmuonworkchain")
    structure = orm.StructureData(cell=[[3, 0, 0], [0, 3, 0], [0, 0, 3]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    structure.append_atom(position=(1.5, 0, 0), symbols="H")
    structure.store()

    results, node = run_get_node(
        module.ImplantMuonWorkChain,
        structure=structure,
        kind_names=orm.List(structure.get_kind_names()),
        implant_muon=False,
        compute_polarization=True,
        undi_code=code,
        undi_metadata={"options": {"resources": {"num_machines": 1}}},
        undi_fields=orm.List([0.0, 1e-3]),
        undi_max_hdims=orm.List([4, 16]),
        undi_max_hdim=orm.Int(16),
    )
    assert node.is_finished_ok
    assert labels_of(node) == ["MultiSiteUndiPolarizationAndKT"]
    multisites = node.called[0]
    assert multisites.process_label == "MultiSitesWorkChain"
    assert multisites.inputs.max_hdim.value == 16  # forwarded from `undi_max_hdim`
    assert labels_of(multisites) == ["polarization_structure_0"]
    assert isinstance(results["polarization"], orm.Dict)
    assert results["polarization"].pk == multisites.outputs.results["site_0"]["KT_task"].pk
