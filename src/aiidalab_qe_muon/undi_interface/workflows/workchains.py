"""WorkChains computing the muon polarization with UNDI and the Kubo-Toyabe model.

The three work chains are nested::

    MultiSitesWorkChain                  one child per muon site
     └─ UndiAndKuboToyabeWorkChain       call link label: ``polarization_structure_{site}``
         ├─ calcfunction                 ``structure_file`` (the structure, as a file for ``undi``)
         ├─ ShellJob (undi.kubo_toyabe)  ``KuboToyabe_run``
         ├─ UndiMultiAnalysisWorkChain   ``convergence_check`` (B = 0, all ``max_hdims``; first site only)
         │   └─ ShellJob (undi)          ``iter_{i}``
         └─ UndiMultiAnalysisWorkChain   ``undi_runs`` (all ``B_mods``, the production ``max_hdim``)
             └─ ShellJob (undi)          ``iter_{i}``

The jobs are ``aiida_shell.ShellJob``: the ``code`` is a python with ``undi[undi]`` installed, and the jobs run
``python -m undi`` and ``python -m undi.kubo_toyabe`` (see ``undi_interface/calculations/shelljobs.py``).

The call link labels are read by the results viewer: do not change them lightly.
"""

from __future__ import annotations

import copy

from aiida import orm
from aiida.common.links import LinkType
from aiida.engine import WorkChain

from aiidalab_qe_muon.undi_interface.calculations.shelljobs import (
    prepare_kt_inputs,
    prepare_undi_run_inputs,
    structure_to_file,
)

# Hilbert-space dimensions of the convergence check (the largest one is the reference of the check)...
DEFAULT_MAX_HDIMS = [10**2, 10**4, 10**6]
# ... and the one of the production runs, at every field.
DEFAULT_MAX_HDIM = 5 * 10**5

# The metadata of the jobs, if none is given. The OpenMP threads of undi follow the resources of the jobs (see
# `shelljobs.omp_threads`): one CPU, so one thread, if the resources are not given.
DEFAULT_TASK_METADATA: dict = {}


def _get_shelljob_class():
    from aiida.plugins import CalculationFactory

    return CalculationFactory("core.shell")


def _with_call_link_label(inputs: dict, label: str) -> dict:
    """Return the ``ShellJob`` inputs with ``metadata.call_link_label`` set to ``label``."""
    inputs = dict(inputs)
    inputs["metadata"] = {**(inputs.get("metadata") or {}), "call_link_label": label}
    return inputs


def _flatten(nested: dict, prefix: str = "") -> dict:
    """Flatten a nested dictionary into one with dotted keys."""
    flat = {}
    for key, value in nested.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}."))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _returned_results(process: orm.ProcessNode) -> dict:
    """Return the nodes returned in the ``results`` namespace of a finished child work chain, with dotted keys."""
    nested = process.base.links.get_outgoing(link_type=LinkType.RETURN).nested()
    return _flatten(nested.get("results", {}))


def _add_undi_inputs(spec, *, B_mods, max_hdims, production_max_hdim=None):
    """Add to ``spec`` the inputs shared by the work chains.

    ``production_max_hdim`` adds the ``max_hdim`` input, if not ``None``: the size used by the production runs.
    """
    spec.input(
        "code",
        valid_type=orm.AbstractCode,
        help="The code of the python (with `undi` installed) that runs the UNDI and Kubo-Toyabe calculations, as `aiida-shell` jobs.",
    )
    spec.input(
        "B_mods",
        valid_type=orm.List,
        default=lambda: orm.List(list(B_mods)),
        help="The magnetic fields (in Tesla) for which the polarization is computed.",
    )
    spec.input(
        "max_hdims",
        valid_type=orm.List,
        default=lambda: orm.List(list(max_hdims)),
        help="The maximum Hilbert-space dimensions of the convergence check (at zero field).",
    )
    if production_max_hdim is not None:
        spec.input(
            "max_hdim",
            valid_type=orm.Int,
            default=lambda: orm.Int(production_max_hdim),
            help="The maximum Hilbert-space dimension of the production runs, at every field.",
        )
    spec.input(
        "atom_as_muon",
        valid_type=orm.Str,
        default=lambda: orm.Str("H"),
        help="The element in the structure that stands for the muon.",
    )
    spec.input(
        "algorithm",
        valid_type=orm.Str,
        default=lambda: orm.Str("fast"),
        help="The UNDI algorithm.",
    )
    spec.input(
        "angular_integration_steps",
        valid_type=orm.Int,
        default=lambda: orm.Int(7),
        help="The number of steps of the angular integration.",
    )
    spec.input(
        "single_precision",
        valid_type=orm.Bool,
        default=lambda: orm.Bool(True),
        help="Run the C++ algorithm of undi in single precision: about half the memory and faster, "
        "with a small loss of accuracy.",
    )
    spec.input(
        "task_metadata",
        valid_type=dict,
        non_db=True,
        required=False,
        help="The `metadata` (e.g. `options`, `resources`) of every `ShellJob`. The OpenMP threads of undi are the "
        "CPUs of the `resources` (unless `OMP_NUM_THREADS` is set in the options).",
    )


class UndiMultiAnalysisWorkChain(WorkChain):
    """Run UNDI in parallel for every combination of magnetic field and Hilbert-space dimension."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            "structure_file",
            valid_type=orm.SinglefileData,
            help="The structure with the muon, as a file `undi` can read (see `structure_to_file`).",
        )
        _add_undi_inputs(spec, B_mods=[0.0], max_hdims=[1e1])
        spec.input(
            "convergence_check",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Run UNDI in the mode used to check the convergence with `max_hdims`.",
        )
        spec.outline(cls.setup, cls.run_undi, cls.inspect_undi)
        spec.output_namespace(
            "results",
            dynamic=True,
            required=False,
            help="The UNDI results of each run, labelled `iter_{i}`, with `i` running over `B_mods` first, then `max_hdims`.",
        )
        spec.exit_code(
            401, "ERROR_UNDI_RUN_FAILED", message="At least one UNDI job failed."
        )

    def setup(self):
        self.ctx.task_metadata = copy.deepcopy(
            self.inputs.get("task_metadata", DEFAULT_TASK_METADATA)
        )
        self.ctx.n_runs = 0

    def run_undi(self):
        """Submit all the UNDI runs; they are independent, so they run in parallel."""
        ShellJob = _get_shelljob_class()
        index = 0
        for B_mod in self.inputs.B_mods.get_list():
            for max_hdim in self.inputs.max_hdims.get_list():
                label = f"iter_{index}"
                inputs = prepare_undi_run_inputs(
                    structure_file=self.inputs.structure_file,
                    code=self.inputs.code,
                    B_mod=B_mod,
                    max_hdim=max_hdim,
                    atom_as_muon=self.inputs.atom_as_muon.value,
                    convergence_check=self.inputs.convergence_check.value,
                    algorithm=self.inputs.algorithm.value,
                    angular_integration_steps=self.inputs.angular_integration_steps.value,
                    single_precision=self.inputs.single_precision.value,
                    metadata=self.ctx.task_metadata,
                )
                future = self.submit(ShellJob, **_with_call_link_label(inputs, label))
                self.report(f"submitted UNDI run `{label}` <PK={future.pk}>")
                self.to_context(**{label: future})
                index += 1
        self.ctx.n_runs = index

    def inspect_undi(self):
        """Fail if any run failed, otherwise return the results."""
        children = {f"iter_{i}": self.ctx[f"iter_{i}"] for i in range(self.ctx.n_runs)}
        failed = {label: child for label, child in children.items() if not child.is_finished_ok}
        for label, child in failed.items():
            self.report(f"UNDI run `{label}` <PK={child.pk}> failed with exit status {child.exit_status}")
        if failed:
            return self.exit_codes.ERROR_UNDI_RUN_FAILED

        for label, child in children.items():
            self.out(f"results.{label}", child.outputs.results)


class UndiAndKuboToyabeWorkChain(WorkChain):
    """Compute the Kubo-Toyabe polarization and the UNDI polarization (with optional convergence check) of one structure."""

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input("structure", valid_type=orm.StructureData, help="The structure with the muon.")
        _add_undi_inputs(
            spec,
            B_mods=[0.0],
            max_hdims=DEFAULT_MAX_HDIMS,
            production_max_hdim=DEFAULT_MAX_HDIM,
        )
        spec.input(
            "convergence_check",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Also run UNDI at zero field for all the `max_hdims`, to check the convergence.",
        )
        spec.outline(
            cls.setup,
            cls.run_analyses,
            cls.inspect_analyses,
        )
        spec.output(
            "results.KT_task",
            valid_type=orm.Data,
            help="The Kubo-Toyabe polarization.",
        )
        spec.output_namespace(
            "results.undi_conv_task",
            dynamic=True,
            required=False,
            help="The UNDI results of the convergence check (only if `convergence_check`).",
        )
        spec.output_namespace(
            "results.undi_task",
            dynamic=True,
            required=False,
            help="The UNDI results at each field, for the production `max_hdim`.",
        )
        spec.exit_code(401, "ERROR_KT_FAILED", message="The Kubo-Toyabe job failed.")
        spec.exit_code(
            402,
            "ERROR_CONVERGENCE_FAILED",
            message="The UNDI convergence-check work chain failed.",
        )
        spec.exit_code(403, "ERROR_UNDI_FAILED", message="The UNDI work chain failed.")

    def setup(self):
        self.ctx.task_metadata = copy.deepcopy(
            self.inputs.get("task_metadata", DEFAULT_TASK_METADATA)
        )
        # the jobs read the structure from a file: made once, shared by all of them.
        self.ctx.structure_file = structure_to_file(
            self.inputs.structure, metadata={"call_link_label": "structure_file"}
        )

    def _undi_inputs(self, **overrides):
        inputs = {
            "structure_file": self.ctx.structure_file,
            "code": self.inputs.code,
            "B_mods": self.inputs.B_mods,
            "max_hdims": self.inputs.max_hdims,
            "atom_as_muon": self.inputs.atom_as_muon,
            "algorithm": self.inputs.algorithm,
            "angular_integration_steps": self.inputs.angular_integration_steps,
            "single_precision": self.inputs.single_precision,
            "task_metadata": self.ctx.task_metadata,
        }
        inputs.update(overrides)
        return inputs

    def run_analyses(self):
        """Submit the Kubo-Toyabe job and the UNDI work chains; they are independent, so they run in parallel."""
        ShellJob = _get_shelljob_class()

        kt_inputs = prepare_kt_inputs(
            structure_file=self.ctx.structure_file,
            code=self.inputs.code,
            metadata=self.ctx.task_metadata,
        )
        kt_future = self.submit(ShellJob, **_with_call_link_label(kt_inputs, "KuboToyabe_run"))
        self.report(f"submitted Kubo-Toyabe job <PK={kt_future.pk}>")
        self.to_context(kt_task=kt_future)

        if self.inputs.convergence_check.value:
            future = self.submit(
                UndiMultiAnalysisWorkChain,
                **self._undi_inputs(
                    B_mods=orm.List([0.0]),
                    convergence_check=orm.Bool(True),
                    metadata={"call_link_label": "convergence_check"},
                ),
            )
            self.report(f"submitted UNDI convergence check <PK={future.pk}>")
            self.to_context(undi_conv_task=future)

        # The production runs: every field, with the production Hilbert-space dimension only.
        future = self.submit(
            UndiMultiAnalysisWorkChain,
            **self._undi_inputs(
                max_hdims=orm.List([self.inputs.max_hdim.value]),
                convergence_check=orm.Bool(False),
                metadata={"call_link_label": "undi_runs"},
            ),
        )
        self.report(f"submitted UNDI runs <PK={future.pk}>")
        self.to_context(undi_task=future)

    def inspect_analyses(self):
        """Fail if any child failed, otherwise return the results."""
        kt_task = self.ctx.kt_task
        if not kt_task.is_finished_ok:
            self.report(f"the Kubo-Toyabe job <PK={kt_task.pk}> failed")
            return self.exit_codes.ERROR_KT_FAILED

        if self.inputs.convergence_check.value:
            undi_conv_task = self.ctx.undi_conv_task
            if not undi_conv_task.is_finished_ok:
                self.report(f"the UNDI convergence check <PK={undi_conv_task.pk}> failed")
                return self.exit_codes.ERROR_CONVERGENCE_FAILED

        undi_task = self.ctx.undi_task
        if not undi_task.is_finished_ok:
            self.report(f"the UNDI runs <PK={undi_task.pk}> failed")
            return self.exit_codes.ERROR_UNDI_FAILED

        self.out("results.KT_task", kt_task.outputs.results)
        if self.inputs.convergence_check.value:
            for key, node in _returned_results(self.ctx.undi_conv_task).items():
                self.out(f"results.undi_conv_task.{key}", node)
        for key, node in _returned_results(undi_task).items():
            self.out(f"results.undi_task.{key}", node)


class MultiSitesWorkChain(WorkChain):
    """Compute the polarization of all the candidate muon sites.

    The convergence check is run for the first site only.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input_namespace(
            "structure_group",
            valid_type=orm.StructureData,
            dynamic=True,
            help="The structures with the muon, one per site, labelled by the site index.",
        )
        _add_undi_inputs(
            spec,
            B_mods=[0, 2e-3, 4e-3, 6e-3, 8e-3],
            max_hdims=DEFAULT_MAX_HDIMS,
            production_max_hdim=DEFAULT_MAX_HDIM,
        )
        spec.outline(cls.setup, cls.run_sites, cls.inspect_sites)
        spec.output_namespace(
            "results",
            dynamic=True,
            required=False,
            help="The results of each site, in the `site_{index}` namespaces; see `UndiAndKuboToyabeWorkChain`.",
        )
        spec.exit_code(401, "ERROR_SITE_FAILED", message="The polarization of at least one site failed.")

    def setup(self):
        self.ctx.task_metadata = copy.deepcopy(
            self.inputs.get("task_metadata", DEFAULT_TASK_METADATA)
        )
        self.ctx.site_indexes = list(self.inputs.structure_group.keys())

    def run_sites(self):
        """Submit one `UndiAndKuboToyabeWorkChain` per site; they run in parallel."""
        for position, site_index in enumerate(self.ctx.site_indexes):
            future = self.submit(
                UndiAndKuboToyabeWorkChain,
                structure=self.inputs.structure_group[site_index],
                code=self.inputs.code,
                B_mods=self.inputs.B_mods,
                max_hdims=self.inputs.max_hdims,
                max_hdim=self.inputs.max_hdim,
                atom_as_muon=self.inputs.atom_as_muon,
                algorithm=self.inputs.algorithm,
                angular_integration_steps=self.inputs.angular_integration_steps,
                single_precision=self.inputs.single_precision,
                task_metadata=self.ctx.task_metadata,
                convergence_check=orm.Bool(position == 0),
                metadata={"call_link_label": f"polarization_structure_{site_index}"},
            )
            self.report(f"submitted polarization of site {site_index} <PK={future.pk}>")
            self.to_context(**{f"site_{site_index}": future})

    def inspect_sites(self):
        """Fail if any site failed, otherwise return the results."""
        children = {site_index: self.ctx[f"site_{site_index}"] for site_index in self.ctx.site_indexes}
        failed = {i: child for i, child in children.items() if not child.is_finished_ok}
        for site_index, child in failed.items():
            self.report(f"polarization of site {site_index} <PK={child.pk}> failed with exit status {child.exit_status}")
        if failed:
            return self.exit_codes.ERROR_SITE_FAILED

        for site_index, child in children.items():
            for key, node in _returned_results(child).items():
                self.out(f"results.site_{site_index}.{key}", node)
