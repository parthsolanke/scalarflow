import math
import unittest

import numpy as np

from scalarflow.engine import value
from scalarflow.nn import layer, module, network, neuron


class ModuleTests(unittest.TestCase):
    def test_base_module_has_no_parameters(self):
        self.assertEqual(module().parameters(), [])

    def test_base_module_zero_grad_is_noop(self):
        module().zero_grad()


class NeuronTests(unittest.TestCase):
    def test_parameter_count(self):
        n = neuron(3)
        self.assertEqual(len(n.parameters()), 4)
        self.assertEqual(n.parameters(), n.weights + [n.bias])

    def test_forward_tanh(self):
        n = neuron(2, "tanh")
        n.weights = [value(1), value(-2)]
        n.bias = value(0.5)

        out = n([1, 2])

        self.assertIsInstance(out, value)
        self.assertAlmostEqual(out.data, math.tanh(-2.5))

    def test_forward_sigmoid(self):
        n = neuron(1, "sigmoid")
        n.weights = [value(2)]
        n.bias = value(-1)

        out = n([1])

        self.assertAlmostEqual(out.data, 1 / (1 + math.exp(-1)))

    def test_forward_relu(self):
        n = neuron(1, "relu")
        n.weights = [value(1)]
        n.bias = value(-2)

        self.assertEqual(n([1]).data, 0)

    def test_default_activation_is_tanh(self):
        n = neuron(1)
        n.weights = [value(1)]
        n.bias = value(0)
        self.assertAlmostEqual(n([0.5]).data, math.tanh(0.5))


class LayerTests(unittest.TestCase):
    def test_parameter_count(self):
        l = layer(3, 2)
        self.assertEqual(len(l.parameters()), 2 * (3 + 1))

    def test_single_neuron_returns_scalar(self):
        l = layer(1, 1)
        self.assertIsInstance(l([value(1)]), value)

    def test_multiple_neurons_return_list(self):
        l = layer(2, 3)
        out = l([value(1), value(2)])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)


class NetworkTests(unittest.TestCase):
    def test_parameter_count(self):
        net = network(2, [2, 1])
        self.assertEqual(len(net.parameters()), 2 * (2 + 1) + 1 * (2 + 1))

    def test_forward_returns_value(self):
        net = network(2, [2, 1])
        self.assertIsInstance(net([value(0), value(1)]), value)

    def test_predict_returns_floats(self):
        net = network(2, [1])
        predictions = net.predict([[0, 0], [1, 1]])
        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertIsInstance(prediction, float)

    def test_zero_grad_resets_all_parameters(self):
        net = network(2, [2, 1])
        for param in net.parameters():
            param.gradient = 5
        net.zero_grad()
        self.assertTrue(all(param.gradient == 0 for param in net.parameters()))

    def test_mse_loss_backprops_gradients(self):
        net = network(2, [1])
        loss = net.mse_loss([[0, 0], [1, 1]], [[0], [1]])

        self.assertIsInstance(loss, value)
        loss.backward()
        self.assertTrue(any(param.gradient != 0 for param in net.parameters()))

    def test_svm_loss_returns_value_and_accuracy(self):
        net = network(2, [1], "tanh")
        loss, accuracy = net.svm_loss([[0, 0], [1, 1]], [1, -1])

        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

        loss.backward()
        self.assertTrue(any(param.gradient != 0 for param in net.parameters()))

    def test_svm_loss_with_batch_size(self):
        net = network(2, [1], "tanh")
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([1, 1, -1, -1])
        loss, accuracy = net.svm_loss(x, y, batch_size=2)

        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_fit_updates_parameters(self):
        net = network(2, [2, 1], "tanh")
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        outputs = [[0], [1], [1], [0]]
        before = [param.data for param in net.parameters()]

        net.fit(inputs, outputs, epochs=3)

        after = [param.data for param in net.parameters()]
        self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
