"""AiiDAlab QE plugin for Muon spectroscopy."""

from aiidalab_qe.plugins import QeAppPlugin


class MuonPlugin(QeAppPlugin):
    @property
    def outline(self) -> str:
        return "Muon spectroscopy"

    def configuration(self):
        from aiidalab_qe_muon.app.configuration.model import (
            MuonConfigurationSettingsModel,
        )
        from aiidalab_qe_muon.app.configuration.view import (
            MuonConfigurationSettingPanel,
        )

        return {
            "panel": MuonConfigurationSettingPanel,
            "model": MuonConfigurationSettingsModel,
        }

    def resources(self):
        from aiidalab_qe_muon.app.codes.mvc import (
            MuonResourceSettingsModel,
            MuonResourcesSettingsPanel,
        )

        return {
            "panel": MuonResourcesSettingsPanel,
            "model": MuonResourceSettingsModel,
        }

    def result(self):
        from aiidalab_qe_muon.app.results.model import MuonResultsModel
        from aiidalab_qe_muon.app.results.view import MuonResultsPanel

        return {
            "panel": MuonResultsPanel,
            "model": MuonResultsModel,
        }

    def workchain(self):
        from aiidalab_qe_muon.app.workchain import workchain_and_builder

        return workchain_and_builder

    def importers(self):
        from aiidalab_qe_muon.app.structure_importer.structure import ImportMagnetism

        return (ImportMagnetism,)

    def guides(self):
        from pathlib import Path

        return {
            "title": self.outline,
            "path": Path(__file__).resolve().parent / "guides",
        }
