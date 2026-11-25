import random
import unittest

from scalarflow.engine import value
from scalarflow.nn import Model, layer, module, neuron


class TestNeuron(unittest.TestCase):

    def test_parameters(self):
        n = neuron(3)

        params = n.parameters()

        self.assertEqual(len(params), 4)
        self.assertIn(n.bias, params)

    def test_call_returns_value(self):
        n = neuron(2)

        out = n([1, 2])

        self.assertIsInstance(out, value)

    def test_activation_selection(self):
        random.seed(0)
        n = neuron(2, _activation='relu')

        out = n([1, 2])

        self.assertGreaterEqual(out.data, 0)


class TestLayer(unittest.TestCase):

    def test_neuron_count(self):
        l = layer(2, 3)

        self.assertEqual(len(l.neurons), 3)
        self.assertEqual(len(l.parameters()), 9)

    def test_single_output_unwrapped(self):
        l = layer(2, 1)

        out = l([1, 2])

        self.assertIsInstance(out, value)

    def test_multiple_output_is_list(self):
        l = layer(2, 3)

        out = l([1, 2])

        self.assertIsInstance(out, list)
        self.assertEqual(len(out), 3)


class TestModel(unittest.TestCase):

    def test_layers_and_parameters(self):
        model = Model(2, [4, 2, 1])

        self.assertEqual(len(model.layers), 3)
        # (2+1)*4 + (4+1)*2 + (2+1)*1 = 12 + 10 + 3
        self.assertEqual(len(model.parameters()), 25)

    def test_call_returns_value(self):
        model = Model(2, [3, 1])

        out = model([1, 2])

        self.assertIsInstance(out, value)

    def test_predict_returns_floats(self):
        model = Model(2, [3, 1])

        predictions = model.predict([[0, 0], [1, 1]])

        self.assertEqual(len(predictions), 2)
        for prediction in predictions:
            self.assertIsInstance(prediction, float)

    def test_module_base_parameters_empty(self):
        m = module()

        self.assertEqual(m.parameters(), [])

    def test_zero_grad(self):
        model = Model(2, [2, 1])
        for param in model.parameters():
            param.gradient = 5

        model.zero_grad()

        self.assertTrue(all(param.gradient == 0 for param in model.parameters()))

    def test_mse_loss_returns_value(self):
        model = Model(2, [3, 1])

        loss = model.mse_loss([[0, 0], [1, 1]], [0, 1])

        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(loss.data, 0)

    def test_fit_runs(self):
        model = Model(2, [3, 1])

        model.fit([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0], epochs=1)

        self.assertEqual(len(model.parameters()), 13)


if __name__ == '__main__':
    unittest.main()
