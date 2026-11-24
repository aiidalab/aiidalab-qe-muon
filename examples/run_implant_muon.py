"""Example: launch the ImplantMuonWorkChain without the app.

    python run_implant_muon.py --pw-code pw-7.3@cluster --dry-run            # build and validate only
    python run_implant_muon.py --pw-code pw-7.3@cluster                      # polarization only (cheap)
    python run_implant_muon.py --pw-code pw-7.3@cluster --findmuon           # muon sites, then polarization

* Default, polarization only: the structure is Cu with the muon (`H`) as the last site, and only the UNDI /
  Kubo-Toyabe part runs (`compute_findmuon=False`).
* ``--findmuon``: the structure is Cu without the muon; the muon sites are searched with DFT (a minimal 1x1x1
  supercell, for a quick test) and the polarization is computed for each unique site.

Even for the polarization only, the builder is created through ``FindMuonWorkChain.get_builder_from_protocol``, so a
``pw`` code and an installed pseudo family are needed (``aiida-pseudo install sssp``), although no DFT is run.

The work chain is run in this process by default (no daemon needed, but a long wait with ``--findmuon``);
use ``--submit`` to send it to the daemon and ``--show <PK>`` to print the results afterwards.
"""

from __future__ import annotations

import argparse

from _common import (
    DEFAULT_FIELDS_MT,
    DEFAULT_MAX_HDIMS,
    PYTHONJOB_CODE,
    cu_h_structure,
    cu_structure,
    print_results,
    undi_task_metadata,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pw-code", help="pw.x code label (required, except with --show)")
    parser.add_argument("--pp-code", default=None, help="pp.x code label (only needed for magnetic systems)")
    parser.add_argument("--undi-code", default=PYTHONJOB_CODE, help="PythonJob code label (default: %(default)s)")
    parser.add_argument("--profile", default=None, help="AiiDA profile (default: the default profile)")
    parser.add_argument("--pseudo-family", default="SSSP/1.3/PBE/efficiency", help="default: %(default)s")
    parser.add_argument("--findmuon", action="store_true", help="also search the muon sites with DFT")
    parser.add_argument(
        "--fields", type=float, nargs="+", default=DEFAULT_FIELDS_MT, help="magnetic fields in mT (default: %(default)s)"
    )
    parser.add_argument(
        "--max-hdims",
        type=int,
        nargs="+",
        default=DEFAULT_MAX_HDIMS,
        help="maximum Hilbert-space dimensions; at least two (default: %(default)s)",
    )
    parser.add_argument("--num-cpus", type=int, default=1, help="MPI processes per pw.x / PythonJob (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="build and validate the builder, do not launch")
    parser.add_argument("--submit", action="store_true", help="submit to the daemon instead of running here")
    parser.add_argument("--show", type=int, metavar="PK", help="print the results of an existing work chain and exit")
    args = parser.parse_args(argv)
    if args.show is None and not args.pw_code:
        parser.error("--pw-code is required (unless --show is used)")
    if len(args.max_hdims) < 2:
        parser.error("--max-hdims needs at least two values (the UNDI runs use the second-to-last one)")
    return args


def get_builder(args, pw_code, pp_code, undi_code):
    """The builder, set up as the app does (see ``aiidalab_qe_muon.app.workchain.get_builder``)."""
    from aiidalab_qe_muon.workflows.implantmuonworkchain import ImplantMuonWorkChain

    dft_options = {
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": args.num_cpus},
        "max_wallclock_seconds": 2 * 3600,
    }
    # the relaxations and the final scf with the muon at the origin share the same scheduler options.
    overrides = {
        "base_relax": {"pw": {"metadata": {"options": dft_options}}},
        "pwscf": {"pw": {"metadata": {"options": dft_options}}},
    }

    return ImplantMuonWorkChain.get_builder_from_protocol(
        pw_muons_code=pw_code,
        pp_code=pp_code,
        undi_code=undi_code,
        undi_metadata=undi_task_metadata(args.num_cpus),
        pseudo_family=args.pseudo_family,
        structure=cu_structure() if args.findmuon else cu_h_structure("octahedral"),
        protocol="fast",
        overrides=overrides,
        compute_findmuon=args.findmuon,
        compute_polarization_undi=True,
        undi_fields=[field * 1e-3 for field in args.fields],  # mT -> T
        undi_max_hdims=args.max_hdims,
        # a minimal supercell, for a quick test: increase it (or pass None to converge it) for real results.
        sc_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        mu_spacing=1.5,
        charge_supercell=True,
        hubbard=False,
        spin_pol_dft=False,  # Cu is not magnetic: no pp code needed.
        gamma_pre_relax=True,
        pre_clustering=True,
    )


def validate(builder):
    """Validate the builder against the process spec, as `submit` would; return the error message, if any."""
    from aiidalab_qe_muon.workflows.implantmuonworkchain import ImplantMuonWorkChain

    error = ImplantMuonWorkChain.spec().inputs.validate(builder._inputs(prune=True))
    return None if error is None else str(error)


def main(argv=None):
    args = parse_args(argv)

    from aiida import load_profile, orm

    load_profile(args.profile)

    if args.show is not None:
        print_results(orm.load_node(args.show))
        return

    from aiida.engine import run_get_node, submit

    pw_code = orm.load_code(args.pw_code)
    pp_code = orm.load_code(args.pp_code) if args.pp_code else None
    undi_code = orm.load_code(args.undi_code)

    builder = get_builder(args, pw_code, pp_code, undi_code)

    print("Builder:")
    print(f"  structure: {builder.structure.get_formula()} ({'host only' if args.findmuon else 'with the muon'})")
    print(f"  implant_muon: {builder.implant_muon.value if hasattr(builder.implant_muon, 'value') else builder.implant_muon}")
    print(f"  undi_fields [T]: {builder.undi_fields.get_list()}")
    print(f"  undi_max_hdims: {builder.undi_max_hdims.get_list()}")
    error = validate(builder)
    print(f"  validation: {'OK' if error is None else error}")

    if args.dry_run or error is not None:
        return

    if args.submit:
        node = submit(builder)
        print(f"ImplantMuonWorkChain submitted, PK: {node.pk}")
        print(f"Monitor with:  verdi process status {node.pk}")
        print(f"Results with:  python {__file__.split('/')[-1]} --show {node.pk}")
    else:
        _, node = run_get_node(builder)
        print_results(node)


if __name__ == "__main__":
    main()
