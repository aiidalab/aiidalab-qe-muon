"""The results viewer on the provenance of real shell jobs: chains run with ``ShellJob`` (a stand-in for ``undi``),
then ``PolarizationModel.fetch_data`` reads the jobs, the way the app does after a calculation."""

import sys

import pytest
import traitlets as tl
from _stubs import fake_modules
from aiida import orm
from aiida.common.extendeddicts import AttributeDict
from aiida.common.links import LinkType
from aiida.engine import run_get_node
from test_shelljobs import code, make_code, stand_in_undi  # noqa: F401  (fixtures)

pytest.importorskip("aiida_shell")

from aiidalab_qe_muon.undi_interface.workflows.workchains import MultiSitesWorkChain  # noqa: E402


@pytest.fixture
def polarization_model(monkeypatch):
    """The model of the polarization plots, with the base class of the app (a stand-in for it, if not installed)."""
    fake_modules(monkeypatch, "aiidalab_qe", "aiidalab_qe.plugins", "aiidalab_qe.common", "aiidalab_qe.common.mvc")
    mvc = sys.modules["aiidalab_qe.common.mvc"]
    if getattr(mvc, "__file__", None) is None:  # the stand-in module
        monkeypatch.setattr(mvc, "Model", tl.HasTraits, raising=False)
    from aiidalab_qe_muon.app.results.sub_mvc.undimodel import PolarizationModel

    return PolarizationModel


def cu_h(distance):
    structure = orm.StructureData(cell=[[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    structure.append_atom(position=(0, 0, 0), symbols="Cu")
    structure.append_atom(position=(distance, 0, 0), symbols="H")
    return structure.store()


@pytest.fixture
def multisites(code):  # noqa: F811
    """`MultiSitesWorkChain` on two sites, with real jobs: convergence check (first site only) and production runs."""
    results, node = run_get_node(
        MultiSitesWorkChain,
        structure_group={"3": cu_h(1.5), "7": cu_h(2.0)},
        code=code,
        B_mods=orm.List([0.0, 2e-3]),
        max_hdims=orm.List([4, 16]),
        max_hdim=orm.Int(16),
    )
    assert node.is_finished_ok, node.exit_message
    return results, node


def test_chain_of_shell_jobs(multisites):
    results, node = multisites
    assert sorted(results["results"]) == ["site_3", "site_7"]
    site = next(
        link.node
        for link in node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()
        if link.link_label == "polarization_structure_3"
    )
    children = {
        link.link_label: link.node
        for link in site.base.links.get_outgoing().all()
        if link.link_type in (LinkType.CALL_CALC, LinkType.CALL_WORK)
    }
    # the structure file (once per site), the KT job, the convergence check (first site) and the production runs
    assert set(children) == {"structure_file", "KuboToyabe_run", "convergence_check", "undi_runs"}
    assert children["structure_file"].process_label == "structure_to_file"
    assert isinstance(children["KuboToyabe_run"], orm.CalcJobNode)
    assert children["KuboToyabe_run"].process_type == "aiida.calculations:core.shell"
    # every UNDI job of the production runs got the production size
    assert [job.inputs.nodes.max_hdim.value for job in children["undi_runs"].called] == [16, 16]
    # the second site has no convergence check
    other = next(
        link.node
        for link in node.base.links.get_outgoing(link_type=LinkType.CALL_WORK).all()
        if link.link_label == "polarization_structure_7"
    )
    assert "convergence_check" not in {link.link_label for link in other.base.links.get_outgoing().all()}


def test_viewer_reads_the_plots_of_all_sites(multisites, polarization_model):
    """Plot mode: the production runs of every site, and the Kubo-Toyabe output."""
    results, _ = multisites
    model = polarization_model(mode="plot")
    # `ImplantMuonWorkChain.polarization`: the KT of one site, from which the viewer finds all the sites
    model.muon = AttributeDict({"polarization": results["results"]["site_3"]["KT_task"]})
    model.fetch_data()

    assert sorted(model.muons) == ["3", "7"]
    assert model.fields == [0.0, 2.0]  # mT
    assert model.max_hdims == [16, 16]
    assert model.production_max_hdim == 16
    for muon in model.muons.values():
        assert muon.fields == [0.0, 2.0]
        assert [run[0]["B_ext"] for run in muon.results] == [0.0, 2e-3]
        assert set(muon.KT_output) >= {"t", "KT"}
    assert model.isotopes == [[{"Cu": 63}, {"Cu": 1.5}, 1.0]]


def test_viewer_reads_the_convergence_of_the_first_site_only(multisites, polarization_model):
    """Analysis mode: the convergence check is run for the first site only, at zero field, for every max_hdim."""
    results, _ = multisites
    model = polarization_model(mode="analysis")
    model.muon = AttributeDict({"polarization": results["results"]["site_7"]["KT_task"]})  # any site leads to all the others
    model.fetch_data()

    assert list(model.muons) == ["3"]
    assert model.max_hdims == [4, 16]
    assert model.production_max_hdim == 16
    assert [run[0]["max_hdim"] for run in model.muons["3"].results] == [4, 16]
    assert [run[0]["convergence_check"] for run in model.muons["3"].results] == [True, True]
