import sys
from types import SimpleNamespace

import pytest
from aiida.orm import load_code
from click.testing import CliRunner

from aiidalab_qe_muon import __main__ as plugin_cli


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
    monkeypatch.setattr(plugin_cli, "_computers", lambda: [SimpleNamespace(label="localhost")])
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
    monkeypatch.setattr(plugin_cli, "_computers", lambda: [])
    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3", "--computer", "nowhere"])
    assert result.exit_code != 0 and "computer `nowhere` does not exist" in result.output


def test_setup_does_nothing_if_the_code_exists(monkeypatch):
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)
    monkeypatch.setattr(plugin_cli, "load_code", lambda label: SimpleNamespace())
    monkeypatch.setattr(plugin_cli, "load_computer", lambda label: SimpleNamespace(label=label))
    monkeypatch.setattr("aiida.plugins.CalculationFactory", lambda name: object)
    monkeypatch.setattr(plugin_cli, "InstalledCode", FakeInstalledCode)
    FakeInstalledCode.created = []

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0 and not FakeInstalledCode.created
    assert "already installed" in result.output


def _computers_in_profile(monkeypatch, labels):
    """A profile with these computers only (`load_computer` raises for the others)."""
    from aiida.common.exceptions import NotExistent

    computers = {label: SimpleNamespace(label=label) for label in labels}

    def load_computer(label):
        try:
            return computers[label]
        except KeyError:
            raise NotExistent(label)

    monkeypatch.setattr(plugin_cli, "load_computer", load_computer)
    monkeypatch.setattr(plugin_cli, "_computers", lambda: list(computers.values()))


def test_setup_uses_the_only_computer_if_there_is_no_localhost(fake_setup, monkeypatch):
    """A deployment with only a HyperQueue computer: the code goes there, and the message says so."""
    _computers_in_profile(monkeypatch, ["hq"])

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code == 0, result.output
    assert fake_setup == ["python3-shell@hq"]
    assert FakeInstalledCode.created[0]["computer"].label == "hq"
    assert "using the only one of the profile, `hq`" in result.output


def test_setup_prefers_localhost_over_the_other_computers(fake_setup, monkeypatch):
    _computers_in_profile(monkeypatch, ["hq", "localhost"])

    assert CliRunner().invoke(plugin_cli.cli, ["setup-python3"]).exit_code == 0
    assert fake_setup == ["python3-shell@localhost"]


def test_setup_lists_the_computers_if_the_choice_is_ambiguous(fake_setup, monkeypatch):
    _computers_in_profile(monkeypatch, ["hq", "cluster"])

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])

    assert result.exit_code != 0
    assert "hq, cluster" in result.output and "--computer" in result.output
    assert not FakeInstalledCode.created


def test_setup_explicit_computer_is_not_replaced_by_another(fake_setup, monkeypatch):
    _computers_in_profile(monkeypatch, ["hq"])

    result = CliRunner().invoke(plugin_cli.cli, ["setup-python3", "--computer", "nowhere"])

    assert result.exit_code != 0
    assert "`nowhere` does not exist" in result.output and "available: hq" in result.output


def test_setup_on_a_real_profile(aiida_localhost, monkeypatch):
    """The real `InstalledCode`: created once, the second setup does nothing."""
    pytest.importorskip("aiida_shell")
    monkeypatch.setattr(plugin_cli, "load_profile", lambda *args: None)  # the profile is loaded by the fixture

    first = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert first.exit_code == 0, first.output
    assert "created" in first.output
    code = load_code("python3-shell@localhost")
    assert str(code.filepath_executable) == sys.executable
    assert code.default_calc_job_plugin == "core.shell"

    again = CliRunner().invoke(plugin_cli.cli, ["setup-python3"])
    assert again.exit_code == 0 and "already installed" in again.output
