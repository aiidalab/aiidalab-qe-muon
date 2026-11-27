"""Tests of the quantities of the convergence analysis: ΔP = P - Pᵣ and 100·ΔP/Pᵣ (no app needed)."""

from types import SimpleNamespace

import numpy as np
import pytest
from _stubs import fake_modules
from numpy.testing import assert_allclose

from aiidalab_qe_muon.utils.convergence import CONVERGENCE_QUANTITIES, convergence_quantity, format_hdim


def test_numbers_of_the_report():
    """P = 0.8, Pᵣ = 1: ΔP = -0.2 and 100·ΔP/Pᵣ = -20 %."""
    assert convergence_quantity(0.8, 1.0, "deltaP") == pytest.approx(-0.2)
    assert convergence_quantity(0.8, 1.0, "deltaP_rel") == pytest.approx(-20.0)
    assert convergence_quantity(0.8, 1.0, "P") == pytest.approx(0.8)


def test_relative_is_in_percent_of_the_reference():
    p, p_ref = np.array([0.9, 0.45, -0.3]), np.array([1.0, 0.5, -0.25])
    # the relative value has the sign of ΔP/Pᵣ: ΔP = -0.05 over a negative reference is +20 %
    assert convergence_quantity(p, p_ref, "deltaP_rel") == pytest.approx([-10.0, -10.0, 20.0])
    assert convergence_quantity(p, p_ref, "deltaP") == pytest.approx([-0.1, -0.05, -0.05])


def test_reference_against_itself_is_zero():
    p_ref = np.array([1.0, 0.5, 0.1])
    assert convergence_quantity(p_ref, p_ref, "deltaP") == pytest.approx(0.0)
    assert convergence_quantity(p_ref, p_ref, "deltaP_rel") == pytest.approx(0.0)


def test_relative_is_nan_where_the_reference_is_zero():
    """P_r(t) crossing zero must not give inf (the point is left out of the plot), nor a warning."""
    with np.errstate(all="raise"):
        values = convergence_quantity([0.2, 0.1, 0.4], [1.0, 0.0, 0.5], "deltaP_rel")
    assert values[0] == pytest.approx(-80.0)
    assert np.isnan(values[1])
    assert values[2] == pytest.approx(-20.0)


def test_unknown_quantity():
    with pytest.raises(ValueError, match="Unknown convergence quantity"):
        convergence_quantity(1.0, 1.0, "bad")


def test_every_toggle_value_has_a_label():
    assert set(CONVERGENCE_QUANTITIES) == {"P", "deltaP", "deltaP_rel"}


@pytest.fixture
def undiwidget(monkeypatch):
    """The `undiwidget` module: with the real app if installed, else with a stand-in for `aiidalab_qe`."""
    fake_modules(
        monkeypatch,
        "aiidalab_qe",
        "aiidalab_qe.common",
        "aiidalab_qe.common.widgets",
        "aiidalab_qe.common.infobox",
        "aiidalab_qe.common.mvc",
        "aiidalab_qe.plugins",
    )
    from aiidalab_qe_muon.app.results.sub_mvc import undiwidget

    return undiwidget


@pytest.mark.parametrize(
    "kind, expected_y, ylabel",
    [
        ("P", [[1.0, 0.8], [1.0, 1.0]], "P(t)"),
        ("deltaP", [[0.0, -0.2], [0.0, 0.0]], "ΔP(t)"),
        ("deltaP_rel", [[0.0, -20.0], [0.0, 0.0]], "100·ΔP(t)/Pᵣ(t) (%)"),
    ],
)
def test_widget_plots_the_quantity_of_the_label(undiwidget, kind, expected_y, ylabel):
    """What is sent to the plot: two max_hdim curves, the reference being the largest one."""
    import plotly.graph_objects as go

    muon = SimpleNamespace(
        results=[[{"B_ext": 0.0}], [{"B_ext": 0.0}]],
        data={"x": [0.0, 1.0], "y": {"lf": [{"signal_z": [1.0, 0.8]}, {"signal_z": [1.0, 1.0]}]}},
        KT_output=None,
    )
    # `_update_plot` only needs these attributes: call it with a stand-in, not with a whole widget.
    widget = SimpleNamespace(
        fig=go.Figure(),
        rendered=True,
        _on_add_KT_change=lambda: None,
        _model=SimpleNamespace(
            mode="analysis",
            directions="z",
            field_direction="lf",
            max_hdims=[10**2, 10**4],
            muons={"0": muon},
            selected_indexes=["0"],
            selected_labels=["A"],
            plotting_quantity=kind,
            plot_KT=False,
        ),
    )
    undiwidget.UndiPlotWidget._update_plot(widget)

    assert len(widget.fig.data) == 2
    assert_allclose([list(trace.y) for trace in widget.fig.data], expected_y, atol=1e-12)
    assert widget.fig.layout.yaxis.title.text == ylabel


@pytest.mark.parametrize(
    ("value", "html"),
    [
        (100, "10<sup>2</sup>"),
        (10**6, "10<sup>6</sup>"),
        (5 * 10**5, "5&middot;10<sup>5</sup>"),
        (1e4, "10<sup>4</sup>"),
        (123456, "123456"),
    ],
)
def test_format_hdim(value, html):
    """5e5 was shown as 10^5 (the floor of its log10)."""
    assert format_hdim(value) == html
