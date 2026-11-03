"""JSON bridge between the ScalarFlow HTTP server and the Python library.

Reads one JSON request from stdin and writes one JSON response to stdout.
The server runs this as a short-lived subprocess for each API call.
"""

import contextlib
import io
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scalarflow.nn import network  # noqa: E402


VALID_ACTIVATIONS = {"", "sigmoid", "tanh", "relu"}


def _activation(value):
    activation = value or ""
    if activation not in VALID_ACTIVATIONS:
        raise ValueError(
            "activation must be one of: '', 'sigmoid', 'tanh', 'relu'"
        )
    return activation


def _serialize(model):
    return {
        "num_in": len(model.layers[0].neurons[0].weights),
        "layer_sizes": [len(layer.neurons) for layer in model.layers],
        "activation": model.unit_activation,
        "layers": [
            {
                "neurons": [
                    {
                        "weights": [float(w.data) for w in neuron.weights],
                        "bias": float(neuron.bias.data),
                        "activation": neuron._activation,
                    }
                    for neuron in layer.neurons
                ]
            }
            for layer in model.layers
        ],
    }


def _build(spec):
    num_in = int(spec["num_in"])
    layer_sizes = [int(size) for size in spec["layer_sizes"]]
    activation = _activation(spec.get("activation"))
    if num_in < 1 or not layer_sizes or any(size < 1 for size in layer_sizes):
        raise ValueError("num_in and every layer size must be positive integers")
    return network(num_in, layer_sizes, activation)


def _load(spec):
    model = _build(spec)
    layers = spec.get("layers")
    if not isinstance(layers, list) or len(layers) != len(model.layers):
        raise ValueError("model.layers does not match layer_sizes")

    for layer_spec, layer in zip(layers, model.layers):
        neurons = layer_spec.get("neurons")
        if not isinstance(neurons, list) or len(neurons) != len(layer.neurons):
            raise ValueError("model neuron count does not match layer_sizes")
        for neuron_spec, neuron in zip(neurons, layer.neurons):
            weights = neuron_spec.get("weights")
            if not isinstance(weights, list) or len(weights) != len(neuron.weights):
                raise ValueError("model weight count does not match its inputs")
            for weight, new_value in zip(neuron.weights, weights):
                weight.data = float(new_value)
            neuron.bias.data = float(neuron_spec.get("bias", 0.0))
            neuron._activation = _activation(
                neuron_spec.get("activation", model.unit_activation)
            )
    return model


def _check_inputs(inputs, expected_width):
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("inputs must be a non-empty array of rows")
    for row in inputs:
        if not isinstance(row, list) or len(row) != expected_width:
            raise ValueError(
                "each input row must contain exactly %d numbers" % expected_width
            )


def _stats(model):
    layers = len(model.layers)
    neurons = sum(len(layer.neurons) for layer in model.layers)
    weights = sum(len(neuron.weights) for layer in model.layers for neuron in layer.neurons)
    return {
        "layers": layers,
        "neurons": neurons,
        "weights": weights,
        "biases": neurons,
        "parameters": weights + neurons,
    }


def _train(request):
    if request.get("model") is not None:
        model = _load(request["model"])
    else:
        model = _build(request)

    inputs = request.get("inputs")
    outputs = request.get("outputs")
    expected_width = len(model.layers[0].neurons[0].weights)
    _check_inputs(inputs, expected_width)

    if not isinstance(outputs, list) or len(outputs) != len(inputs):
        raise ValueError("outputs must contain one target per input row")
    targets = [float(target) for target in outputs]

    epochs = int(request.get("epochs") or 100)
    if epochs < 1:
        raise ValueError("epochs must be a positive integer")

    # network.fit prints progress; keep stdout clean for the JSON response.
    with contextlib.redirect_stdout(io.StringIO()):
        model.fit(inputs, targets, epochs)

    loss = float(model.mse_loss(inputs, targets).data)
    return {
        "model": _serialize(model),
        "predictions": [float(prediction) for prediction in model.predict(inputs)],
        "loss": loss,
    }


def _predict(request):
    model = _load(request["model"])
    inputs = request.get("inputs")
    expected_width = len(model.layers[0].neurons[0].weights)
    _check_inputs(inputs, expected_width)
    return {
        "predictions": [float(prediction) for prediction in model.predict(inputs)]
    }


def _summary(request):
    model = _load(request["model"])
    return {"stats": _stats(model), "model": _serialize(model)}


HANDLERS = {
    "train": _train,
    "predict": _predict,
    "summary": _summary,
}


def main():
    try:
        request = json.loads(sys.stdin.read() or "{}")
        action = request.get("action")
        if action not in HANDLERS:
            raise ValueError("action must be one of: train, predict, summary")
        response = HANDLERS[action](request)
        response["ok"] = True
    except ValueError as exc:
        response = {"ok": False, "error": str(exc), "kind": "validation"}
    except Exception as exc:  # pragma: no cover - surfaced to the HTTP caller
        response = {"ok": False, "error": str(exc), "kind": "internal"}

    try:
        json.dump(response, sys.stdout, allow_nan=False)
        sys.stdout.write("\n")
    except ValueError:
        json.dump(
            {"ok": False, "error": "model produced a non-finite value", "kind": "internal"},
            sys.stdout,
        )
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
