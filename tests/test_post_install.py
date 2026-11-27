from types import SimpleNamespace

import pytest
from click.testing import CliRunner

from aiidalab_qe_muon import __main__ as plugin_cli


def test_verify_post_install_passes(monkeypatch, tmp_path):
    executable = tmp_path / "python3"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)
    monkeypatch.setattr(plugin_cli, "load_profile", lambda _profile: None)
    monkeypatch.setattr(
        plugin_cli,
        "load_code",
        lambda _label: SimpleNamespace(filepath_executable=str(executable)),
    )

    result = CliRunner().invoke(plugin_cli.cli, ["verify-post-install"])

    assert result.exit_code == 0
    assert "setup is valid" in result.output


def test_verify_post_install_fails_when_code_is_missing(monkeypatch):
    from aiida.common.exceptions import NotExistent

    monkeypatch.setattr(plugin_cli, "load_profile", lambda _profile: None)

    def missing_code(_label):
        raise NotExistent("missing")

    monkeypatch.setattr(plugin_cli, "load_code", missing_code)

    result = CliRunner().invoke(plugin_cli.cli, ["verify-post-install"])

    assert result.exit_code != 0
    assert "python3-shell@localhost is not configured" in result.output


@pytest.mark.parametrize(
    ("path_state", "expected_message"),
    [
        ("unset", "python3-shell@localhost executable is not set"),
        ("missing", "python3-shell@localhost executable is missing"),
        ("directory", "python3-shell@localhost executable is not a file"),
        ("not_executable", "python3-shell@localhost executable is not executable"),
    ],
)
def test_verify_post_install_reports_executable_path_state(
    monkeypatch, tmp_path, path_state, expected_message
):
    executable = tmp_path / "python3"
    if path_state == "unset":
        filepath = None
    elif path_state == "directory":
        executable.mkdir()
        filepath = str(executable)
    elif path_state == "not_executable":
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        executable.chmod(0o644)
        filepath = str(executable)
    else:
        filepath = str(executable)
    monkeypatch.setattr(plugin_cli, "load_profile", lambda _profile: None)
    monkeypatch.setattr(
        plugin_cli,
        "load_code",
        lambda _label: SimpleNamespace(filepath_executable=filepath),
    )

    result = CliRunner().invoke(plugin_cli.cli, ["verify-post-install"])

    assert result.exit_code != 0
    assert expected_message in result.output


def test_setup_creates_a_shell_code_that_does_not_clash_with_the_pythonjob_one(monkeypatch):
    """The code runs undi through aiida-shell: `core.shell`, labelled `python3-shell`."""
    from aiida.common.exceptions import NotExistent

    loaded, commands = [], []

    def missing_code(label):
        loaded.append(label)
        raise NotExistent("missing")

    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", missing_code)
    monkeypatch.setattr(plugin_cli.subprocess, "run", lambda command, check: commands.append(command))

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0
    assert loaded == ["python3-shell@localhost"]  # a `python3@localhost` of the pythonjob version is not in the way
    (command,) = commands
    assert command[command.index("--label") + 1] == "python3-shell"
    assert command[command.index("--default-calc-job-plugin") + 1] == "core.shell"


def test_setup_does_nothing_if_the_code_exists(monkeypatch):
    commands = []
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", lambda label: SimpleNamespace())
    monkeypatch.setattr(plugin_cli.subprocess, "run", lambda command, check: commands.append(command))

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0 and not commands
    assert "already installed" in result.output
