import typing as t

from aiida_workgraph import spec, task

from aiidalab_qe_muon.undi_interface.calculations.pythonjobs import (
    compute_KT,
    undi_run,
)


def _metadata_with_label(metadata, label):
    """Return task metadata with a stable call-link label."""
    if hasattr(metadata, "get_dict"):
        metadata = metadata.get_dict()
    metadata = dict(metadata or {})
    metadata["call_link_label"] = label
    return metadata


@task.graph
def multiple_undi_analysis(
    structure,
    B_mods: t.List[t.Union[float, int]] = [0.0],
    atom_as_muon: str = "H",
    max_hdims: t.List[t.Union[float, int]] = [1e1],
    convergence_check: bool = False,
    algorithm: str = "fast",
    angular_integration_steps: int = 7,
    code=None,
    task_metadata={"options": {"custom_scheduler_commands": "export OMP_NUM_THREADS=1"}},
) -> spec.namespace(results=spec.dynamic(t.Any)):
    """Build parallel UNDI analyses for all fields and Hilbert-space sizes."""
    results = {}
    index = 0
    for B_mod in B_mods:
        for max_hdim in max_hdims:
            output = undi_run(
                structure=structure,
                B_mod=B_mod,
                max_hdim=max_hdim,
                atom_as_muon=atom_as_muon,
                convergence_check=convergence_check,
                algorithm=algorithm,
                angular_integration_steps=angular_integration_steps,
                metadata=_metadata_with_label(task_metadata, f"iter_{index}"),
                code=code,
                register_pickle_by_value=True,
            )
            results[f"iter_{index}"] = output.results
            index += 1

    return {"results": results}


@task.graph
def UndiAndKuboToyabe(
    structure,
    B_mods: t.List[t.Union[float, int]] = [0.0],
    atom_as_muon: str = "H",
    max_hdims: t.List[t.Union[float, int]] = [1e1],
    convergence_check: bool = False,
    algorithm: str = "fast",
    angular_integration_steps: int = 7,
    code=None,
    task_metadata={"options": {"custom_scheduler_commands": "export OMP_NUM_THREADS=1"}},
) -> spec.namespace(
    results=spec.namespace(
        KT_task=t.Any,
        undi_conv_task=spec.dynamic(t.Any),
        undi_task=spec.dynamic(t.Any),
    )
):
    """Build the UNDI and Kubo-Toyabe polarization analyses."""
    results = {
        "KT_task": compute_KT(
            structure=structure,
            code=code,
            metadata=_metadata_with_label(task_metadata, "KuboToyabe_run"),
            register_pickle_by_value=True,
        ).results
    }

    if convergence_check:
        results["undi_conv_task"] = multiple_undi_analysis(
            structure=structure,
            B_mods=[0.0],
            max_hdims=max_hdims,
            atom_as_muon=atom_as_muon,
            convergence_check=convergence_check,
            algorithm=algorithm,
            angular_integration_steps=angular_integration_steps,
            code=code,
            task_metadata=task_metadata,
            metadata={"call_link_label": "convergence_check"},
        ).results
    else:
        results["undi_conv_task"] = {}

    results["undi_task"] = multiple_undi_analysis(
        structure=structure,
        B_mods=B_mods,
        max_hdims=max_hdims[-2:-1],
        atom_as_muon=atom_as_muon,
        convergence_check=False,
        algorithm=algorithm,
        angular_integration_steps=angular_integration_steps,
        code=code,
        task_metadata=task_metadata,
        metadata={"call_link_label": "undi_runs"},
    ).results

    return {"results": results}


@task.graph
def MultiSites(
    structure_group: t.Annotated[dict, spec.dynamic(t.Any)],
    code=None,
    B_mods: t.List[t.Union[float, int]] = [0, 2e-3, 4e-3, 6e-3, 8e-3],
    max_hdims: t.List[t.Union[float, int]] = [10**2, 10**4, 10**6, 10**8],
    task_metadata={"options": {"custom_scheduler_commands": "export OMP_NUM_THREADS=1"}},
) -> spec.namespace(
    results=spec.dynamic(
        spec.namespace(
            KT_task=t.Any,
            undi_conv_task=spec.dynamic(t.Any),
            undi_task=spec.dynamic(t.Any),
        )
    )
):
    """Build polarization analyses for all candidate muon sites."""
    results = {}
    for index, (site_index, structure) in enumerate(structure_group.items()):
        output = UndiAndKuboToyabe(
            structure=structure,
            B_mods=B_mods,
            max_hdims=max_hdims,
            convergence_check=index == 0,
            algorithm="fast",
            code=code,
            task_metadata=task_metadata,
            metadata={
                "call_link_label": f"polarization_structure_{site_index}"
            },
        )
        results[f"site_{site_index}"] = output.results

    return {"results": results}
