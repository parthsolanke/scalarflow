import math
import random

from scalarflow.nn import layer, network, neuron


def test_neuron_parameters_and_output():
    random.seed(0)
    n = neuron(3, _activation="tanh")

    assert len(n.parameters()) == 4

    out = n([0.1, 0.2, 0.3])
    assert isinstance(out.data, float)
    assert not math.isnan(out.data)


def test_layer_returns_scalar_for_single_unit():
    random.seed(0)
    single = layer(2, 1)
    multi = layer(2, 3)

    assert not isinstance(single([0.1, 0.2]), list)
    assert isinstance(multi([0.1, 0.2]), list)
    assert len(multi([0.1, 0.2])) == 3
    assert len(multi.parameters()) == 3 * 3


def test_network_parameter_count_and_predict_shape():
    random.seed(0)
    model = network(num_in=2, layer_sizes=[3, 1])

    # first layer: 3 neurons * (2 weights + 1 bias) = 9
    # second layer: 1 neuron * (3 weights + 1 bias) = 4
    assert len(model.parameters()) == 13

    predictions = model.predict([[0.0, 0.0], [1.0, 1.0]])
    assert len(predictions) == 2
    assert all(isinstance(p, float) for p in predictions)


def test_zero_grad_resets_parameters():
    random.seed(0)
    model = network(num_in=2, layer_sizes=[2, 1])

    loss = model.mse_loss([[0.0, 0.0]], [1.0])
    loss.backward()
    assert any(p.gradient != 0 for p in model.parameters())

    model.zero_grad()
    assert all(p.gradient == 0 for p in model.parameters())


def test_fit_reduces_loss():
    random.seed(0)
    model = network(num_in=1, layer_sizes=[1], unit_activation="tanh")
    inputs = [[0.0]]
    outputs = [0.5]

    before = model.mse_loss(inputs, outputs).data
    model.fit(inputs, outputs, epochs=100)
    after = model.mse_loss(inputs, outputs).data

    assert after < before


def test_summary_prints_architecture(capsys):
    random.seed(0)
    model = network(num_in=2, layer_sizes=[2, 1])

    model.summary()

    captured = capsys.readouterr()
    assert "Network architecture" in captured.out
    assert "Total number of parameters: 9" in captured.out
