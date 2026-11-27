# aiidalab-qe-muon
Plugin to compute muon stopping sites and related properties via the aiida-muon and aiida-musconv AiiDA plugins

## Installation

In order to load this plugin into QeApp, you need to switch to the `main` branch for your `aiidalab-qe`.

Then, install this plugin by:

```shell
pip install -e .
```

### The code that runs undi

The polarization is computed by running the command line of [undi](https://undi.readthedocs.io/en/latest/index.html)
(`python -m undi` and `python -m undi.kubo_toyabe`) as [aiida-shell](https://aiida-shell.readthedocs.io) jobs. They need a
`python` code, with the `core.shell` plugin, whose environment has `undi` installed (with its `undi` extra: `ase`, `tqdm`).

On the local computer, to set up the code with the python of the environment that runs the command (the one with `undi`;
it is labelled `python3-shell@localhost`, so it does not clash with the `python3` code of the former `aiida-pythonjob`
version; use `--label` and `--computer` for other names. Without `--computer`, it uses `localhost`, or the only
computer of the profile if there is no `localhost`):

```shell
aiidalab-qe-muon setup-python3
```

On a computer (local or remote) already set up in AiiDA, you can create the environment and the code by hand. For example,
with conda:

```shell
conda create -n undi python=3.12 -y
conda activate undi
pip install "undi[undi] @ git+https://github.com/mikibonacci/undi.git@shelljobs"
```

and then, giving the path of the python of that environment:

```shell
verdi code create core.code.installed \
    --label python3-shell \
    --computer <computer_label> \
    --default-calc-job-plugin core.shell \
    --filepath-executable <path/to/envs/undi/bin/python> \
    --non-interactive
```

(use `--prepend-text` to load modules or activate the environment if needed).

## License

MIT

## Contact

miki.bonacci@psi.ch


## Acknowledgements
We acknowledge support from:
* the [NCCR MARVEL](http://nccr-marvel.ch/) funded by the Swiss National Science Foundation;
* the PNRR MUR project [ECS-00000033-ECOSISTER](https://ecosister.it/);

<img src="https://raw.githubusercontent.com/positivemuon/aiida-muon/main/docs/source/images/MARVEL_logo.png" width="250px" height="131px"/>
<img src="https://raw.githubusercontent.com/positivemuon/aiida-muon/main/docs/source/images/ecosister_logo.png" width="300px" height="84px"/>
