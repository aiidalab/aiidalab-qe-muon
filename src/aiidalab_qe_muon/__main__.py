import sys

import click
from aiida import load_profile
from aiida.common.exceptions import MissingEntryPointError, NotExistent
from aiida.orm import InstalledCode, load_code, load_computer

"""
Automatic installation of the code that runs undi: the python of this environment, as a `core.shell` code
(`aiida-shell`), labelled `python3-shell` so that it does not clash with a `python3` code of the former
`aiida-pythonjob` version of this plugin.

The python needs `undi[undi]` (and its dependencies) installed. So we only setup in AiiDA.
"""

CODE_LABEL = "python3-shell"
COMPUTER_LABEL = "localhost"


@click.group()
def cli():
    pass


def _computers():
    """The computers of the profile."""
    from aiida.orm import Computer

    return Computer.collection.all()


def _resolve_computer(computer):
    """The computer to put the code on.

    An explicit `--computer` must exist. Without it, `localhost` is used; if there is no `localhost` (e.g. a deployment
    with only a HyperQueue computer), the only computer of the profile is used. Anything else is ambiguous: we say which
    computers there are.
    """
    if computer is not None:
        try:
            return load_computer(computer)
        except NotExistent as error:
            labels = ", ".join(c.label for c in _computers()) or "none"
            raise click.ClickException(
                f"The computer `{computer}` does not exist in this AiiDA profile (available: {labels})."
            ) from error

    try:
        return load_computer(COMPUTER_LABEL)
    except NotExistent:
        pass
    computers = _computers()
    if len(computers) == 1:
        click.echo(f"No computer `{COMPUTER_LABEL}`: using the only one of the profile, `{computers[0].label}`.")
        return computers[0]
    labels = ", ".join(c.label for c in computers) or "none"
    raise click.ClickException(
        f"No computer `{COMPUTER_LABEL}` in this AiiDA profile (available: {labels}): choose one with `--computer`."
    )


@cli.command(help="Setup the python3-shell code (python of this environment) in the current AiiDA profile.")
@click.option("--label", default=CODE_LABEL, show_default=True, help="Label of the code.")
@click.option(
    "--computer",
    default=None,
    help=f"Label of the computer. Default: `{COMPUTER_LABEL}`, or the only computer of the profile if there is no "
    f"`{COMPUTER_LABEL}`.",
)
def setup_python3(label, computer):
    """Create the code of the python of this environment, as `core.shell` code.

    It uses `sys.executable`, i.e. the python that runs this command (the one of the environment where the plugin and
    `undi` are installed), and not the first `python3` of the PATH. The code is created through the AiiDA API: nothing
    depends on `verdi` being in the PATH of the process that runs this command (the plugin manager of the app).
    """
    load_profile()

    from aiida.plugins import CalculationFactory

    try:
        CalculationFactory("core.shell")
    except MissingEntryPointError as error:
        raise click.ClickException(
            "The `core.shell` calculation is not available: install `aiida-shell` in this environment "
            f"({sys.executable} -m pip install aiida-shell)."
        ) from error

    aiida_computer = _resolve_computer(computer)
    identifier = f"{label}@{aiida_computer.label}"
    try:
        load_code(identifier)
    except NotExistent:
        pass
    else:
        click.echo(f"Code {identifier} is already installed! Nothing to do here.")
        return

    InstalledCode(
        label=label,
        computer=aiida_computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="core.shell",
        description="Python (with undi) to run the UNDI and Kubo-Toyabe calculations, through aiida-shell.",
    ).store()
    click.echo(
        f"Code {identifier} created, with the python of this environment: {sys.executable}. "
        "Select it in the app, in the Muon resources (python)."
    )


if __name__ == "__main__":
    cli()
