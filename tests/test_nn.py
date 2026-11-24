import unittest

from scalarflow.engine import value
from scalarflow.nn import layer, network, neuron


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


if __name__ == '__main__':
    unittest.main()
