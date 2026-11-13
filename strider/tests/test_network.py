import numpy as np
import pytest

torch = pytest.importorskip("torch")

from strider.network.coadd import inverse_variance_coadd, relative_inverse_variance  # noqa: E402
from strider.network.matching import log_mean_exp, token_cosine  # noqa: E402
from strider.tests.synthetic_package import toy_network  # noqa: E402


def test_the_coadd_weights_visits_by_inverse_variance():
    flux = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    mask = torch.ones(2, 2)
    log_error = torch.log(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    coadd, error, measured = inverse_variance_coadd(flux, mask, log_error, torch.ones(2))
    torch.testing.assert_close(coadd, (4 * flux[0] + flux[1]) / 5)
    torch.testing.assert_close(error, torch.full((2,), 1 / np.sqrt(1.25)).float())
    assert measured.tolist() == [1.0, 1.0]

    # visits given over their own scales coadd the same, in units of the scales' geometric mean
    scale = torch.tensor([4.0, 1.0])
    scaled, _, _ = inverse_variance_coadd(flux / scale[:, None], mask, log_error - scale.log()[:, None], scale)
    torch.testing.assert_close(scaled * 2, coadd)


def test_relative_inverse_variance():
    error = torch.tensor([[1.0, 2.0, 4.0, 0.0]])
    weight = relative_inverse_variance(error, torch.tensor([[1.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(weight, torch.tensor([[4.0, 1.0, 0.25, 0.0]]))


def test_log_mean_exp():
    values = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.3, 0.4]])
    support = torch.tensor([[True, True, False], [False, False, False]])
    score, valid = log_mean_exp(values, support, 0.1)
    assert score[0] == pytest.approx(0.1 * np.log(np.mean(np.exp(np.array([1.0, 5.0])))), rel=1e-6)
    assert score[1] == 0.0 and valid.tolist() == [True, False]


def test_a_spectrum_matches_itself():
    tokens = torch.randn(1, 8, 4)
    support = torch.ones(1, 8, dtype=torch.bool)
    similarity, valid = token_cosine(
        tokens, support, tokens, support, torch.tensor([True]), 0.5, 0.75, torch.ones(1, 8)
    )
    assert similarity.item() == pytest.approx(1.0) and valid.item()


def test_the_source_detector_summaries():
    from strider.network.source_detector import summaries

    flux = torch.tensor([[1.0, 3.0, 9.0], [3.0, 5.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    found = summaries(flux, mask, torch.tensor([0.0, 50.0]), visit_count_reference=4)
    measured = torch.tensor([1.0, 3.0, 3.0, 5.0, 0.0])
    coherent = torch.tensor([2.0, 4.0, 0.0]).square().mean().sqrt()
    fluctuation = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0]).div(5).sum().sqrt()
    expected = torch.stack(
        [
            torch.tanh(measured.mean() / 5),
            measured.abs().mean().log1p(),
            measured.square().mean().sqrt().log1p(),
            coherent.log1p(),
            fluctuation.log1p(),
            (coherent / fluctuation).log() / 5,
            torch.tensor(0.5).log1p(),
            torch.tensor(0.5),
        ]
    )
    torch.testing.assert_close(found, expected)
    assert summaries(flux, mask, torch.tensor([0.0, 50.0])).shape == (6,)


def toy_inputs(days):
    from strider.preprocess import network_inputs, prepare

    rng = np.random.default_rng(1)
    wavelength = np.geomspace(7400.0, 18300.0, 150)
    flux = [5.0 + day / 10.0 + np.sin(wavelength / 700.0) + rng.normal(size=150) for day in days]
    error = [np.full(150, 1.0) for _ in days]
    grid = {"wavelength_grid": {"minimum": 7500.0, "maximum": 18175.0, "bins": 128}}
    return network_inputs(
        prepare([wavelength] * len(days), flux, error, 60000.0 + np.asarray(days), settings=grid), "cpu"
    )


def test_weights_load_by_name_and_shape():
    network = toy_network()
    names = network.weight_names()
    weights = {names[key]: value for key, value in network.state_dict().items()}
    assert {"roman_reference.log_coadd_scale", "evidence_sufficiency.network.0.weight"} <= weights.keys()
    assert not any("brightness" in name or "sequence" in name for name in weights)
    loaded = toy_network(seed=1)
    loaded.load_weights(weights)
    for model in (network, loaded):
        model.eval().cache_references()
    inputs = toy_inputs([0.0, 10.0, 25.0])
    with torch.no_grad():
        first, second = network(*inputs), loaded(*inputs)
    for key in ("coadd_term", "support", "source_logit"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="unknown .*extra.*missing .*log_coadd_scale"):
        loaded.load_weights({k: v for k, v in weights.items() if "log_coadd" not in k} | {"extra": torch.zeros(1)})
    with pytest.raises(ValueError, match="at another shape"):
        loaded.load_weights(weights | {"roman_reference.window_weight": torch.zeros(1)})


def test_objects_scored_together_score_as_each_alone():
    """Side by side and padded to the most visits, an object's outputs do not depend on the others: the source logit
    to the bit, the coadd term to float32 rounding (the matrix products differ in shape)."""
    network = toy_network().eval()
    network.cache_references()
    objects = [toy_inputs(days) for days in ([0.0, 10.0, 25.0], [5.0], [0.0, 0.0, 3.0, 40.0, 60.0])]
    with torch.no_grad():
        alone = [network(*inputs) for inputs in objects]
        together = network.forward_many(objects[::-1])
    for single, n in zip(alone, (2, 1, 0)):
        assert torch.equal(together["source_logit"][n], single["source_logit"])
        assert torch.equal(together["support"][n], single["support"])
        torch.testing.assert_close(together["coadd_term"][n], single["coadd_term"], rtol=1e-6, atol=1e-6)


def test_every_tensor_is_made_on_the_inputs_device():
    """On a GPU, a tensor made without the inputs' device would land on the CPU. Here new tensors default to the
    "meta" device instead, so any such tensor fails to mix with the inputs."""
    network = toy_network().eval()
    network.cache_references()
    inputs = toy_inputs([0.0, 10.0, 25.0])
    with torch.no_grad():
        expected = network(*inputs)
        with torch.device("meta"):
            found = network.forward_many([inputs, toy_inputs([5.0])])
    torch.testing.assert_close(found["coadd_term"][0], expected["coadd_term"], rtol=1e-6, atol=1e-6)
