import contextlib
import io
import math
import unittest

import numpy as np

from scalarflow.engine import value
from scalarflow.nn import layer, module, network, neuron


class NeuronTests(unittest.TestCase):

    def test_parameters(self):
        unit = neuron(3)
        params = unit.parameters()

        self.assertEqual(len(params), 4)
        self.assertEqual(unit.bias, params[-1])
        self.assertTrue(all(isinstance(p, value) for p in params))

    def test_call_returns_value(self):
        out = neuron(2)([1.0, 2.0])
        self.assertIsInstance(out, value)

    def test_activation_ranges(self):
        sigmoid_out = neuron(2, 'sigmoid')([1.0, -1.0])
        self.assertGreaterEqual(sigmoid_out.data, 0)
        self.assertLessEqual(sigmoid_out.data, 1)

        tanh_out = neuron(2, 'tanh')([1.0, -1.0])
        self.assertGreaterEqual(tanh_out.data, -1)
        self.assertLessEqual(tanh_out.data, 1)

        relu_out = neuron(2, 'relu')([1.0, -1.0])
        self.assertGreaterEqual(relu_out.data, 0)


class LayerTests(unittest.TestCase):

    def test_parameter_count(self):
        dense = layer(3, 4)
        self.assertEqual(len(dense.parameters()), 4 * (3 + 1))

    def test_multi_output_returns_list(self):
        out = layer(2, 3)([1.0, 2.0])
        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)

    def test_single_output_returns_value(self):
        out = layer(2, 1)([1.0, 2.0])
        self.assertIsInstance(out, value)


class NetworkTests(unittest.TestCase):

    def setUp(self):
        self.model = network(2, [4, 2, 1], 'tanh')

    def test_layer_count(self):
        self.assertEqual(len(self.model.layers), 3)

    def test_parameter_count(self):
        # 4*(2+1) + 2*(4+1) + 1*(2+1) = 25
        self.assertEqual(len(self.model.parameters()), 25)

    def test_predict(self):
        predictions = self.model.predict([[0, 0], [1, 1], [1, 0]])
        self.assertEqual(len(predictions), 3)
        self.assertTrue(all(isinstance(p, (int, float)) for p in predictions))

    def test_zero_grad(self):
        for param in self.model.parameters():
            param.gradient = 5
        self.model.zero_grad()
        self.assertTrue(all(p.gradient == 0 for p in self.model.parameters()))

    def test_mse_loss_returns_value(self):
        loss = self.model.mse_loss([[0, 0], [1, 1]], [0, 1])
        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(loss.data, 0)

    def test_svm_loss(self):
        loss, accuracy = network(2, [3, 1], 'tanh').svm_loss(
            [[0, 0], [1, 1]], [-1, 1]
        )
        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_fit_updates_parameters(self):
        model = network(2, [2, 1], 'tanh')
        before = [p.data for p in model.parameters()]
        model.fit([[0, 0], [1, 1]], [0, 1], epochs=2)
        after = [p.data for p in model.parameters()]

        self.assertNotEqual(before, after)


class ModuleTests(unittest.TestCase):

    def test_base_module_has_no_parameters(self):
        base = module()
        self.assertEqual(base.parameters(), [])
        base.zero_grad()  # must not raise


class NeuronActivationTests(unittest.TestCase):

    def _unit(self, weights, bias, activation=''):
        unit = neuron(len(weights), activation)
        unit.weights = [value(w) for w in weights]
        unit.bias = value(bias)
        return unit

    def test_empty_activation_defaults_to_tanh(self):
        unit = self._unit([1, 1], 0, '')
        self.assertAlmostEqual(unit([1, 1]).data, math.tanh(2))

    def test_unknown_activation_defaults_to_tanh(self):
        unit = self._unit([1, 1], 0, 'bogus')
        self.assertAlmostEqual(unit([1, 1]).data, math.tanh(2))

    def test_sigmoid_activation(self):
        unit = self._unit([0], 0, 'sigmoid')
        self.assertAlmostEqual(unit([1]).data, 0.5)

    def test_relu_activation_clamps_negative(self):
        unit = self._unit([-1], 0, 'relu')
        self.assertEqual(unit([1]).data, 0)

    def test_repr_mentions_weight_count_and_activation(self):
        unit = neuron(4, 'relu')
        text = repr(unit)
        self.assertIn('weights: 4', text)
        self.assertIn("'Relu'", text)


class LayerActivationTests(unittest.TestCase):

    def test_parameters_are_all_values(self):
        dense = layer(2, 3)
        self.assertTrue(all(isinstance(p, value) for p in dense.parameters()))

    def test_zero_grad_resets_layer_parameters(self):
        dense = layer(2, 3)
        for param in dense.parameters():
            param.gradient = 9

        dense.zero_grad()

        self.assertTrue(all(p.gradient == 0 for p in dense.parameters()))


class NetworkBehaviorTests(unittest.TestCase):

    def test_call_returns_value(self):
        model = network(2, [3, 1], 'tanh')
        self.assertIsInstance(model([1, 0]), value)

    def test_predict_is_deterministic(self):
        model = network(2, [3, 1], 'tanh')
        self.assertEqual(model.predict([[1, 0]]), model.predict([[1, 0]]))

    def test_fit_with_zero_epochs_leaves_parameters_unchanged(self):
        model = network(2, [2, 1], 'tanh')
        before = [p.data for p in model.parameters()]

        model.fit([[0, 0], [1, 1]], [0, 1], epochs=0)

        self.assertEqual(before, [p.data for p in model.parameters()])

    def test_svm_loss_with_batch_size(self):
        model = network(2, [3, 1], 'tanh')
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([-1, 1, 1, -1])

        loss, accuracy = model.svm_loss(x, y, batch_size=2)

        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_summary_prints_parameter_total(self):
        model = network(2, [3, 1], 'tanh')
        output = io.StringIO()

        with contextlib.redirect_stdout(output):
            model.summary()

        self.assertIn('Total number of parameters: 13', output.getvalue())

    def test_repr_mentions_layer_count(self):
        self.assertIn('2 layers', repr(network(2, [3, 1], 'tanh')))


if __name__ == '__main__':
    unittest.main()
