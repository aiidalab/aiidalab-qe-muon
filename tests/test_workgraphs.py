from aiidalab_qe_muon.undi_interface.workflows.workgraphs import (
    MultiSites,
    UndiAndKuboToyabe,
    multiple_undi_analysis,
)


def test_pythonjob_workgraph_builds():
    structure = "structure"

    leaf = multiple_undi_analysis.build(
        structure=structure,
        B_mods=[0.0],
        max_hdims=[4],
    )
    assert [task.name for task in leaf.tasks if not task.name.startswith("graph_")] == [
        "iter_0"
    ]
    assert leaf.outputs.results._get_keys() == ["iter_0"]

    inner = UndiAndKuboToyabe.build(
        structure=structure,
        B_mods=[0.0],
        max_hdims=[4, 4],
        convergence_check=True,
    )
    assert [
        task.name for task in inner.tasks if not task.name.startswith("graph_")
    ] == ["KuboToyabe_run", "convergence_check", "undi_runs"]
    assert inner.outputs.results._get_keys() == [
        "KT_task",
        "undi_conv_task",
        "undi_task",
    ]

    outer = MultiSites.build(
        structure_group={"0": structure},
        B_mods=[0.0],
        max_hdims=[4, 4],
    )
    assert [task.name for task in outer.tasks if not task.name.startswith("graph_")] == [
        "polarization_structure_0"
    ]
    assert outer.outputs.results._get_keys() == ["site_0"]
