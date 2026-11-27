# Examples: running the muon workflows without the app

| Script | What it runs |
|---|---|
| [`run_undi_workchains.py`](run_undi_workchains.py) | The three UNDI / Kubo-Toyabe work chains on their own: `analysis` (`UndiMultiAnalysisWorkChain`), `kubo-toyabe` (`UndiAndKuboToyabeWorkChain`), `multisites` (`MultiSitesWorkChain`). |
| [`run_implant_muon.py`](run_implant_muon.py) | `ImplantMuonWorkChain`, polarization only (default) or with the DFT muon-site search (`--findmuon`). |
| [`_common.py`](_common.py) | Shared helpers: the Cu / Cu+H test structures, the metadata of the shell jobs, and the printing of the results. |

## Prerequisites

1. An AiiDA profile (`verdi presto` is enough to run in-process; the daemon is only needed for `--submit`).
2. The package installed in editable mode, so the work chains are registered as entry points:

   ```shell
   pip install -e .
   ```

3. The code that runs UNDI: a `python` with the `core.shell` plugin (`aiida-shell`), whose environment has `undi[undi]`
   installed. On the local computer:

   ```shell
   aiidalab-qe-muon setup-python3      # creates python3-shell@localhost, with the python of this environment
   ```

   For another computer, see the main [README](../README.md). The scripts take its label with `--code` / `--undi-code`
   (`_common.py` has the default).
4. For `run_implant_muon.py` only: a `pw.x` code, and the pseudopotentials (`aiida-pseudo install sssp`, for the
   default `SSSP/1.3/PBE/efficiency`). A `pw` code is needed even for the polarization only, because the builder
   is created through `FindMuonWorkChain.get_builder_from_protocol`; nothing is run with it in that case.

## Examples

```shell
# the work chains on their own: run here and print the results
python run_undi_workchains.py analysis
python run_undi_workchains.py kubo-toyabe
python run_undi_workchains.py multisites

# the same, through the daemon, and look at the results later
python run_undi_workchains.py multisites --submit
python run_undi_workchains.py --show <PK>

# ImplantMuonWorkChain: check that the builder is valid, then run the polarization only
python run_implant_muon.py --pw-code <pw-label> --dry-run
python run_implant_muon.py --pw-code <pw-label>

# ... or the whole workflow (muon sites with DFT, then polarization)
python run_implant_muon.py --pw-code <pw-label> --findmuon --submit
```

Run `--help` on any script for all the options (fields in mT, `max_hdims`, `max_hdim`, codes, CPUs, profile).
The fields are given in mT and converted to Tesla, as the app does.

## What the scripts print

After a run, `--show` included, the scripts print the state and the `results` tree of the work chain, then walk the
provenance like the results viewer does: Kubo-Toyabe output → its shell job → site work chain →
`MultiSitesWorkChain` → one work chain per site, with the site index read from the call-link label
(`polarization_structure_{index}`) and the `(B, max_hdim)` of every UNDI run of `convergence_check` and
`undi_runs`. If that walk fails or reaches a different `MultiSitesWorkChain`, the viewer would not find the data
either.

On failure they print the work chain report and the reports of the failed calculations.

## Notes

- The UNDI jobs are not MPI: `--num-cpus` are the CPUs of each job and the OpenMP threads of undi (`OMP_NUM_THREADS`
  follows them). UNDI runs in single precision (faster, a small loss of accuracy); `--double-precision` turns it off.
  The output of every job starts with a line that tells which C++ module runs, with how many threads
  (`[undi] C++ module fast_quantum (...); OpenMP threads: N ...`).
- `--max-hdims` are the Hilbert-space dimensions of the convergence check (first site only, at zero field; the largest is
  the reference), and `--max-hdim` the one of the production runs at every field (not used by `analysis`). The app uses
  `10²`, `10⁴`, `10⁶` and `10⁶`; the examples default to smaller values (`10²`, `10³`, `10⁴` and `10⁴`) to run quickly.
- The structures are small Cu supercells with the muon as an `H` in the last site (octahedral for site `0`,
  tetrahedral for site `1` in the `multisites` example), enough to test the workflow, not to get physical results.
- `run_implant_muon.py` uses a 1x1x1 supercell matrix, to keep a `--findmuon` test short; increase it for real
  calculations.
