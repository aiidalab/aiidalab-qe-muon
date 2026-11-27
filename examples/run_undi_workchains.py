"""Example: run the UNDI / Kubo-Toyabe work chains on their own, without the app.

    python run_undi_workchains.py analysis       # UndiMultiAnalysisWorkChain: UNDI at all B x max_hdim
    python run_undi_workchains.py kubo-toyabe    # UndiAndKuboToyabeWorkChain: KT + convergence check + UNDI
    python run_undi_workchains.py multisites     # MultiSitesWorkChain: the above, for two muon sites

By default the work chain is run in this process (``run_get_node``, no daemon needed) and the results are printed.
With ``--submit`` it is sent to the daemon instead; print the results later with ``--show <PK>``.

Requirements: see README.md (a `core.shell` python code, with `undi[undi]` installed in its environment).
Structure: Cu with the muon (`H`) as the last site, see ``_common.py``.
"""

from __future__ import annotations

import argparse

from _common import (
    DEFAULT_FIELDS_MT,
    DEFAULT_MAX_HDIM,
    DEFAULT_MAX_HDIMS,
    UNDI_CODE,
    cu_h_structure,
    print_results,
    undi_task_metadata,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "chain",
        nargs="?",
        choices=["analysis", "kubo-toyabe", "multisites"],
        help="which work chain to run (not needed with --show)",
    )
    parser.add_argument("--code", default=UNDI_CODE, help="label of the python (undi) `core.shell` code (default: %(default)s)")
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the default profile)")
    parser.add_argument(
        "--fields", type=float, nargs="+", default=DEFAULT_FIELDS_MT, help="magnetic fields in mT (default: %(default)s)"
    )
    parser.add_argument(
        "--max-hdims",
        type=float,
        nargs="+",
        default=DEFAULT_MAX_HDIMS,
        help="maximum Hilbert-space dimensions of the convergence check, at zero field; the largest is the "
        "reference (default: %(default)s)",
    )
    parser.add_argument(
        "--max-hdim",
        type=int,
        default=DEFAULT_MAX_HDIM,
        help="maximum Hilbert-space dimension of the production runs, at every field; not used by "
        "`analysis` (default: %(default)s)",
    )
    parser.add_argument(
        "--num-cpus", type=int, default=1, help="CPUs of each job, used as OpenMP threads by undi (default: %(default)s)"
    )
    parser.add_argument(
        "--double-precision", action="store_true", help="run undi in double precision (default: single precision)"
    )
    parser.add_argument("--submit", action="store_true", help="submit to the daemon instead of running here")
    parser.add_argument("--show", type=int, metavar="PK", help="print the results of an existing work chain and exit")
    args = parser.parse_args(argv)
    if args.show is None and args.chain is None:
        parser.error("choose a work chain, or use --show PK")
    return args


def build_inputs(args, code):
    """Inputs of the chosen work chain; the fields are converted from mT to T."""
    from aiida import orm

    from aiidalab_qe_muon.undi_interface.calculations.shelljobs import structure_to_file

    inputs = {
        "code": code,
        "B_mods": orm.List([field * 1e-3 for field in args.fields]),
        "max_hdims": orm.List(args.max_hdims),
        "task_metadata": undi_task_metadata(args.num_cpus),
        "single_precision": orm.Bool(not args.double_precision),
    }
    if args.chain == "multisites":
        # keys are the site indices; the convergence check is run for the first one only.
        inputs["structure_group"] = {
            "0": cu_h_structure("octahedral"),
            "1": cu_h_structure("tetrahedral"),
        }
    elif args.chain == "analysis":
        # the leaf work chain takes the structure as a file for undi (the others make it themselves)
        inputs["structure_file"] = structure_to_file(cu_h_structure("octahedral"))
    else:
        inputs["structure"] = cu_h_structure("octahedral")
        inputs["convergence_check"] = orm.Bool(True)
    if args.chain != "analysis":
        inputs["max_hdim"] = orm.Int(args.max_hdim)
    return inputs


def main(argv=None):
    args = parse_args(argv)

    from aiida import load_profile, orm

    load_profile(args.profile)

    if args.show is not None:
        print_results(orm.load_node(args.show))
        return

    from aiida.engine import run_get_node, submit

    from aiidalab_qe_muon.undi_interface.workflows.workchains import (
        MultiSitesWorkChain,
        UndiAndKuboToyabeWorkChain,
        UndiMultiAnalysisWorkChain,
    )

    workchain = {
        "analysis": UndiMultiAnalysisWorkChain,
        "kubo-toyabe": UndiAndKuboToyabeWorkChain,
        "multisites": MultiSitesWorkChain,
    }[args.chain]
    inputs = build_inputs(args, orm.load_code(args.code))

    if args.submit:
        node = submit(workchain, **inputs)
        print(f"{workchain.__name__} submitted, PK: {node.pk}")
        print(f"Monitor with:  verdi process status {node.pk}")
        print(f"Results with:  python {__file__.split('/')[-1]} --show {node.pk}")
    else:
        _, node = run_get_node(workchain, **inputs)
        print_results(node)


if __name__ == "__main__":
    main()
