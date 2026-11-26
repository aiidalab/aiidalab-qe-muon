"""Helpers shared by the example launch scripts."""

from __future__ import annotations

from aiida import orm
from aiida.common.links import LinkType

PYTHONJOB_CODE = "python-3.12-devbox"  # can be created by `aiidalab-qe-muon setup-python3` with a different label, check it.

# Small values, for a quick test (fields in mT, as in the app). The app uses [10**2, 10**4, 10**6] for the
# convergence check and 10**6 for the production runs, which can take long.
DEFAULT_FIELDS_MT = [0.0, 2.0]
DEFAULT_MAX_HDIMS = [10**2, 10**3, 10**4]  # convergence check (the largest one is the reference)
DEFAULT_MAX_HDIM = 10**4  # production runs


def cu_structure() -> orm.StructureData:
    """fcc Cu, 1x1x1 conventional cells, without the muon (the input to find the muon sites)."""
    from ase.build import bulk

    return orm.StructureData(ase=bulk("Cu", "fcc", a=3.61, cubic=True).repeat((1, 1, 1)))


def cu_h_structure(site: str = "octahedral") -> orm.StructureData:
    """fcc Cu, 1x1x1 conventional cells, with the muon as the last site (an `H`), as the polarization step needs.

    :param site: ``"octahedral"`` or ``"tetrahedral"`` interstitial site.
    """
    from ase import Atom
    from ase.build import bulk

    a = 3.61
    atoms = bulk("Cu", "fcc", a=a, cubic=True)
    position = {"octahedral": (a / 2, 0.0, 0.0), "tetrahedral": (a / 4, a / 4, a / 4)}[site]
    atoms.append(Atom("H", position=position))
    return orm.StructureData(ase=atoms.repeat((1, 1, 1)))


def undi_task_metadata(num_cpus: int = 1, max_wallclock_seconds: int = 3600) -> dict:
    """The `metadata` of every UNDI / Kubo-Toyabe `PythonJob`."""
    return {
        "options": {
            "resources": {"num_machines": 1, "num_mpiprocs_per_machine": num_cpus},
            "max_wallclock_seconds": max_wallclock_seconds,
            "custom_scheduler_commands": "export OMP_NUM_THREADS=1",
        }
    }


def _returned(node: orm.ProcessNode) -> dict:
    return node.base.links.get_outgoing(link_type=LinkType.RETURN).nested()


def _describe(value) -> str:
    if isinstance(value, orm.Dict):
        return f"Dict, keys: {sorted(value.get_dict())}"
    if isinstance(value, orm.List):
        return f"List of {len(value.get_list())} item(s)"
    return type(value).__name__


def _print_tree(nested: dict, indent: int = 2) -> None:
    for key, value in sorted(nested.items()):
        if isinstance(value, dict):
            print(f"{' ' * indent}{key}/")
            _print_tree(value, indent + 2)
        else:
            print(f"{' ' * indent}{key}: <PK={value.pk}> {_describe(value)}")


def _find_multisites(node: orm.ProcessNode) -> orm.ProcessNode | None:
    """The `MultiSitesWorkChain` node: ``node`` itself, or the one called by an `ImplantMuonWorkChain`."""
    if node.process_label == "MultiSitesWorkChain":
        return node
    for called in node.called:
        if called.process_label == "MultiSitesWorkChain":
            return called
    return None


def check_viewer_path(node: orm.ProcessNode) -> None:
    """Walk the provenance the way the results viewer does and print what it would find.

    The viewer starts from the Kubo-Toyabe output of the first site (the `polarization` output of the
    `ImplantMuonWorkChain`) and goes KT job -> site work chain -> `MultiSitesWorkChain` -> one work chain per site.
    """
    multisites = _find_multisites(node)
    if multisites is None:
        return
    if "polarization" in node.outputs:  # ImplantMuonWorkChain: this is what the viewer starts from
        kt = node.outputs.polarization
    else:
        results = _returned(multisites).get("results", {})
        kt = results[sorted(results)[0]]["KT_task"]
    print("\nProvenance walk of the results viewer (KT output -> KT job -> site -> multisites -> sites):")
    if kt.creator is None:
        print(f"  KT output <PK={kt.pk}> was not created by a calculation: the viewer cannot walk up from it!")
        return
    print(f"  KT output <PK={kt.pk}>, created by <PK={kt.creator.pk}> ({kt.creator.process_label})")
    viewer_multisites = kt.creator.caller.caller
    assert viewer_multisites.pk == multisites.pk, "the viewer would not reach the MultiSitesWorkChain!"
    for site in viewer_multisites.called:
        link_label = site.base.links.get_incoming(link_type=LinkType.CALL_WORK).all()[0].link_label
        print(f"  {link_label} <PK={site.pk}> -> site index {link_label.replace('polarization_structure_', '')}")
        outgoing = site.base.links.get_outgoing()
        for label in ("convergence_check", "undi_runs"):
            if label not in outgoing.all_link_labels():
                continue
            jobs = outgoing.get_node_by_label(label).called
            runs = sorted(
                (job.inputs.function_inputs.B_mod.value, int(job.inputs.function_inputs.max_hdim.value)) for job in jobs
            )
            print(f"    {label}: (B [T], max_hdim) = {runs}")


def print_results(node: orm.ProcessNode) -> None:
    """Print the state and the outputs of one of the work chains and, if relevant, the viewer provenance walk."""
    print(f"\n{node.process_label} <PK={node.pk}>: {node.process_state.value}, exit status {node.exit_status}")
    if not node.is_finished_ok:
        from aiida.cmdline.utils.common import get_calcjob_report, get_workchain_report

        print(get_workchain_report(node, "REPORT"))
        for called in node.called_descendants:
            if called.is_failed and isinstance(called, orm.CalcJobNode):
                print(get_calcjob_report(called))
        print(f"\nInspect with: verdi process status {node.pk}; verdi process report {node.pk}")
        return

    print("Outputs:")
    _print_tree(_returned(node))
    check_viewer_path(node)
