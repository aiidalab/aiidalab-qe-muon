import os
import sys
from pathlib import Path

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


@cli.command(help="Setup python3-shell@localhost in the current AiiDA database.")
@click.option("--label", default=CODE_LABEL, show_default=True, help="Label of the code.")
@click.option("--computer", default=COMPUTER_LABEL, show_default=True, help="Label of the computer.")
def setup_python3(label, computer):
    """Create the code of the python of this environment, as `core.shell` code.

    It uses `sys.executable`, i.e. the python that runs this command (the one of the environment where the plugin and
    `undi` are installed), and not the first `python3` of the PATH. The code is created through the AiiDA API: nothing
    depends on `verdi` being in the PATH of the process that runs this command (the plugin manager of the app).
    """
    load_profile()
    identifier = f"{label}@{computer}"
    try:
        load_code(identifier)
    except NotExistent:
        pass
    else:
        click.echo(f"Code {identifier} is already installed! Nothing to do here.")
        return

    from aiida.plugins import CalculationFactory

    try:
        CalculationFactory("core.shell")
    except MissingEntryPointError as error:
        raise click.ClickException(
            "The `core.shell` calculation is not available: install `aiida-shell` in this environment "
            f"({sys.executable} -m pip install aiida-shell)."
        ) from error

    try:
        aiida_computer = load_computer(computer)
    except NotExistent as error:
        raise click.ClickException(
            f"The computer `{computer}` does not exist in this AiiDA profile: use `--computer` (see `verdi computer list`)."
        ) from error

    InstalledCode(
        label=label,
        computer=aiida_computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="core.shell",
        description="Python (with undi) to run the UNDI and Kubo-Toyabe calculations, through aiida-shell.",
    ).store()
    click.echo(f"Code {identifier} created, with the python of this environment: {sys.executable}")


@cli.command(help="Verify the python3-shell@localhost post-install setup.")
@click.option("-p", "--profile", default=None, help="AiiDA profile name (default: the default profile).")
def verify_post_install(profile):
    load_profile(profile)
    try:
        code = load_code("python3-shell@localhost")
    except NotExistent as error:
        raise click.ClickException(
            "AiiDA code python3-shell@localhost is not configured. Run setup-python3."
        ) from error

    executable = code.filepath_executable
    if not executable:
        raise click.ClickException("AiiDA code python3-shell@localhost executable is not set")

    filepath = Path(executable)
    failure_prefix = "AiiDA code python3-shell@localhost executable is"
    if not filepath.exists():
        raise click.ClickException(f"{failure_prefix} missing: {executable}")
    if not filepath.is_file():
        raise click.ClickException(f"{failure_prefix} not a file: {executable}")
    if not os.access(filepath, os.X_OK):
        raise click.ClickException(f"{failure_prefix} not executable: {executable}")

    click.echo("Muon post-install setup is valid.")


if __name__ == "__main__":
    cli()
