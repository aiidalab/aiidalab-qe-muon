from aiida.common.exceptions import NotExistent
import os
import subprocess
from pathlib import Path
from aiida.orm import load_code
from aiida import load_profile
import shutil
import click

"""
Automatic installation of the pythonjob code, needed to run undi.

So we only setup in AiiDA.
"""


@click.group()
def cli():
    pass


@cli.command(help="Setup python3@localhost in the current AiiDA database.")
def setup_python3():
    load_profile()
    try:
        load_code("python3@localhost")
    except NotExistent:
        # Use shutil.which to find the path of the phonopy executable
        python_path = shutil.which("python3")
        if not python_path:
            raise FileNotFoundError("python3 code is not found in PATH")
        # Construct the command as a list of arguments
        command = [
            "verdi",
            "code",
            "create",
            "core.code.installed",
            "--non-interactive",
            "--label",
            "python3",
            "--default-calc-job-plugin",
            "pythonjob.pythonjob",
            "--computer",
            "localhost",
            "--filepath-executable",
            python_path,
        ]

        # Use subprocess.run to run the command
        subprocess.run(command, check=True)
    else:
        print("Code python3@localhost is already installed! Nothing to do here.")


@cli.command(help="Verify the python3@localhost post-install setup.")
@click.option("-p", "--profile", default="default", help="AiiDA profile name.")
def verify_post_install(profile):
    load_profile(profile)
    try:
        code = load_code("python3@localhost")
    except NotExistent as error:
        raise click.ClickException(
            "AiiDA code python3@localhost is not configured. Run setup-python3."
        ) from error

    executable = code.filepath_executable
    if not executable:
        raise click.ClickException("AiiDA code python3@localhost executable is not set")

    filepath = Path(executable)
    failure_prefix = "AiiDA code python3@localhost executable is"
    if not filepath.exists():
        raise click.ClickException(f"{failure_prefix} missing: {executable}")
    if not filepath.is_file():
        raise click.ClickException(f"{failure_prefix} not a file: {executable}")
    if not os.access(filepath, os.X_OK):
        raise click.ClickException(f"{failure_prefix} not executable: {executable}")

    click.echo("Muon post-install setup is valid.")


if __name__ == "__main__":
    cli()
