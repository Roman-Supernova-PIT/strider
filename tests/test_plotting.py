from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from strider import load_model
from strider.io import load_inputs
from strider.plotting.evidence_map import evidence_map


@pytest.mark.model
def test_evidence_map_stays_focused_on_the_visual_evidence():
    paths = sorted(Path("examples/SN20088677_ou").glob("spectrum_*.csv"))
    data = load_inputs(paths)
    model = load_model("models/strider.pt")
    output = model.classify(
        data.wavelength,
        data.flux,
        data.phase,
        return_joint=True,
        return_inputs=True,
    )

    fig = plt.figure(figsize=(12.2, 9.4))
    evidence_map(fig, out=output, meta=data.metadata)
    map_axes = [axis for axis in fig.axes if axis.get_ylabel() == "class"]
    map_axis = map_axes[0]
    lo, hi = map_axis.get_xlim()

    assert len(map_axes) == 1
    assert lo <= output["redshift"]["z_p05"]
    assert hi >= output["redshift"]["z_p95"]
    figure_text = " ".join(
        [text.get_text() for text in fig.texts]
        + [axis.get_title(loc="left") for axis in fig.axes]
        + [text.get_text() for axis in fig.axes for text in axis.texts]
    )
    assert "SN20088677_ou" in figure_text
    assert "Class: Ia" in figure_text
    assert "Redshift:" in figure_text
    assert "Class probabilities" in figure_text
    assert "Redshift posterior" in figure_text
    assert "raw P(Ia)=" not in figure_text
    plt.close(fig)

    with_context = model.classify(
        data.wavelength,
        data.flux,
        data.phase,
        z_prior=(0.12, 0.10),
        return_joint=True,
        return_inputs=True,
    )
    fig = plt.figure(figsize=(12.2, 9.4))
    evidence_map(fig, out=with_context, meta=data.metadata)
    titles = [axis.get_title(loc="left") for axis in fig.axes]
    posterior_axis = next(
        axis for axis in fig.axes if axis.get_title(loc="left") == "Redshift posterior"
    )
    assert sum(bool(axis.images) for axis in fig.axes) == 2
    assert "Spectra only" in titles
    assert "With prior" in titles
    assert "with prior" in posterior_axis.get_legend_handles_labels()[1]
    plt.close(fig)
