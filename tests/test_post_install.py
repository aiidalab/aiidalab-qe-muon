import sys
from types import SimpleNamespace

import pytest
from aiida.orm import load_code
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


def test_verify_uses_the_default_profile_unless_told_otherwise(monkeypatch):
    """`--profile` defaults to the default profile (it used to be a profile called `default`)."""
    profiles = []
    monkeypatch.setattr(plugin_cli, "load_profile", lambda profile=None: profiles.append(profile))
    monkeypatch.setattr(plugin_cli, "load_code", lambda label: SimpleNamespace(filepath_executable=sys.executable))

    assert CliRunner().invoke(plugin_cli.cli, ["verify-post-install"]).exit_code == 0
    assert CliRunner().invoke(plugin_cli.cli, ["verify-post-install", "-p", "other"]).exit_code == 0
    assert profiles == [None, "other"]


class FakeInstalledCode:
    """Stand-in for `InstalledCode`, to see what the setup creates."""

    created = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def store(self):
        self.created.append(self.kwargs)
        return self


@pytest.fixture
def fake_setup(monkeypatch):
    """No profile: the code does not exist, the computer does, `aiida-shell` is there, the code is created in a list."""
    from aiida.common.exceptions import NotExistent

    FakeInstalledCode.created = []
    loaded = []

    def missing_code(label):
        loaded.append(label)
        raise NotExistent("missing")

    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", missing_code)
    monkeypatch.setattr(plugin_cli, "load_computer", lambda label: SimpleNamespace(label=label))
    monkeypatch.setattr(plugin_cli, "InstalledCode", FakeInstalledCode)
    monkeypatch.setattr("aiida.plugins.CalculationFactory", lambda name: object)
    return loaded


def test_setup_creates_a_shell_code_with_the_python_of_the_environment(fake_setup):
    """`core.shell`, labelled `python3-shell` (no clash with a `python3` pythonjob code), with `sys.executable`."""
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0, result.output
    assert fake_setup == ["python3-shell@localhost"]
    (created,) = FakeInstalledCode.created
    assert created["label"] == "python3-shell"
    assert created["default_calc_job_plugin"] == "core.shell"
    # not the first `python3` of the PATH, which can be another python (the one without undi)
    assert created["filepath_executable"] == sys.executable
    assert created["computer"].label == "localhost"


def test_setup_does_not_need_verdi_nor_the_path(fake_setup, monkeypatch):
    """It was calling `verdi` and `which python3`: the plugin manager of the app does not have the PATH of a terminal."""
    monkeypatch.setenv("PATH", "")
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert result.exit_code == 0, result.output
    assert FakeInstalledCode.created[0]["filepath_executable"] == sys.executable


def test_setup_label_and_computer_can_be_chosen(fake_setup):
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3", "--label", "py", "--computer", "cluster"])
    assert result.exit_code == 0, result.output
    assert fake_setup == ["py@cluster"]
    assert FakeInstalledCode.created[0]["label"] == "py"
    assert FakeInstalledCode.created[0]["computer"].label == "cluster"


def test_setup_says_what_is_missing(fake_setup, monkeypatch):
    from aiida.common.exceptions import MissingEntryPointError, NotExistent

    def no_shell(name):
        raise MissingEntryPointError("no core.shell")

    monkeypatch.setattr("aiida.plugins.CalculationFactory", no_shell)
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert result.exit_code != 0 and "install `aiida-shell`" in result.output
    assert not FakeInstalledCode.created

    monkeypatch.undo()
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", lambda label: (_ for _ in ()).throw(NotExistent("missing")))
    monkeypatch.setattr("aiida.plugins.CalculationFactory", lambda name: object)

    def no_computer(label):
        raise NotExistent("no computer")

    monkeypatch.setattr(plugin_cli, "load_computer", no_computer)
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3", "--computer", "nowhere"])
    assert result.exit_code != 0 and "computer `nowhere` does not exist" in result.output


def test_setup_does_nothing_if_the_code_exists(monkeypatch):
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", lambda label: SimpleNamespace())
    monkeypatch.setattr(plugin_cli, "InstalledCode", FakeInstalledCode)
    FakeInstalledCode.created = []

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0 and not FakeInstalledCode.created
    assert "already installed" in result.output


def test_setup_and_verify_on_a_real_profile(aiida_localhost, monkeypatch):
    """The real `InstalledCode`: created once, found by verify, the second setup does nothing."""
    pytest.importorskip("aiida_shell")
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)  # the profile is loaded by the fixture

    first = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    code = load_code("python3-shell@localhost")
    assert str(code.filepath_executable) == sys.executable
    assert code.default_calc_job_plugin == "core.shell"

    assert CliRunner().invoke(plugin_cli.cli, ["verify-post-install"]).exit_code == 0
    again = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert again.exit_code == 0 and "already installed" in again.output
