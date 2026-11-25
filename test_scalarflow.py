import io
import math
import random
import unittest
from contextlib import redirect_stdout

from scalarflow.engine import value
from scalarflow.nn import layer, network, neuron


class EngineTests(unittest.TestCase):
    def test_add_backward(self):
        a = value(2)
        b = value(3)
        c = a + b
        self.assertEqual(c.data, 5)
        c.backward()
        self.assertEqual(a.gradient, 1)
        self.assertEqual(b.gradient, 1)

    def test_mul_backward(self):
        a = value(2)
        b = value(3)
        d = a * b
        self.assertEqual(d.data, 6)
        d.backward()
        self.assertEqual(a.gradient, 3)
        self.assertEqual(b.gradient, 2)

    def test_sub_and_neg(self):
        a = value(5)
        b = value(3)
        self.assertEqual((a - b).data, 2)
        self.assertEqual((b - a).data, -2)
        self.assertEqual((-a).data, -5)
        self.assertEqual((10 - a).data, 5)

    def test_pow_and_div(self):
        x = value(3)
        y = x ** 2
        self.assertEqual(y.data, 9)
        y.backward()
        self.assertEqual(x.gradient, 6)

        a = value(8)
        b = value(2)
        q = a / b
        self.assertEqual(q.data, 4)
        q.backward()
        self.assertEqual(a.gradient, 0.5)
        self.assertEqual(b.gradient, -2)

    def test_chain_rule(self):
        a = value(2)
        b = value(3)
        z = a * b + a
        z.backward()
        self.assertEqual(z.data, 8)
        self.assertEqual(a.gradient, b.data + 1)
        self.assertEqual(b.gradient, a.data)

    def test_exp(self):
        x = value(1)
        e = x.exp()
        self.assertAlmostEqual(e.data, math.e)
        e.backward()
        self.assertAlmostEqual(x.gradient, math.e)

    def test_activations(self):
        x = value(0)
        s = x.sigmoid()
        self.assertAlmostEqual(s.data, 0.5)
        s.backward()
        self.assertAlmostEqual(x.gradient, 0.25)

        t_in = value(0)
        t = t_in.tanh()
        self.assertAlmostEqual(t.data, 0)
        t.backward()
        self.assertAlmostEqual(t_in.gradient, 1)

        r = value(2)
        self.assertEqual(r.relu().data, 2)

        n = value(-3)
        self.assertEqual(n.relu().data, 0)

    def test_relu_backward(self):
        x = value(2)
        x.relu().backward()
        self.assertEqual(x.gradient, 1)

        y = value(-2)
        y.relu().backward()
        self.assertEqual(y.gradient, 0)

    def test_backward_sets_output_gradient(self):
        a = value(2)
        b = value(3)
        z = a * b
        z.backward()
        self.assertEqual(z.gradient, 1)

    def test_reverse_operators_forward(self):
        a = value(2)
        self.assertEqual((5 + a).data, 7)
        self.assertEqual((5 * a).data, 10)
        self.assertEqual((10 - a).data, 8)
        self.assertEqual((10 / a).data, 5)

    def test_reverse_operators_backward(self):
        a = value(2)
        r = 5 + a
        r.backward()
        self.assertEqual(a.gradient, 1)

        b = value(2)
        m = 5 * b
        m.backward()
        self.assertEqual(b.gradient, 5)

        c = value(3)
        s = 10 - c
        s.backward()
        self.assertEqual(c.gradient, -1)

        d = value(2)
        q = 10 / d
        q.backward()
        self.assertAlmostEqual(d.gradient, -2.5)

    def test_pow_fractional_and_negative(self):
        x = value(4)
        y = x ** 0.5
        self.assertAlmostEqual(y.data, 2)
        y.backward()
        self.assertAlmostEqual(x.gradient, 0.25)

        z = value(2)
        inv = z ** -1
        self.assertAlmostEqual(inv.data, 0.5)
        inv.backward()
        self.assertAlmostEqual(z.gradient, -0.25)

    def test_pow_rejects_non_numeric_exponent(self):
        with self.assertRaises(AssertionError):
            value(2) ** value(3)

    def test_gradient_accumulates_on_shared_node(self):
        a = value(3)
        y = a * a
        self.assertEqual(y.data, 9)
        y.backward()
        self.assertEqual(a.gradient, 6)

    def test_sigmoid_nonzero_gradient(self):
        x = value(2)
        s = x.sigmoid()
        self.assertAlmostEqual(s.data, 1 / (1 + math.exp(-2)))
        s.backward()
        self.assertAlmostEqual(x.gradient, s.data * (1 - s.data))

    def test_tanh_nonzero_gradient(self):
        x = value(0.5)
        t = x.tanh()
        self.assertAlmostEqual(t.data, math.tanh(0.5))
        t.backward()
        self.assertAlmostEqual(x.gradient, 1 - math.tanh(0.5) ** 2)

    def test_exp_of_zero_chain(self):
        x = value(0)
        e = (x * 2).exp()
        self.assertAlmostEqual(e.data, 1)
        e.backward()
        self.assertAlmostEqual(x.gradient, 2)

    def test_relu_at_zero(self):
        x = value(0)
        out = x.relu()
        self.assertEqual(out.data, 0)
        out.backward()
        self.assertEqual(x.gradient, 0)

    def test_trace_graph(self):
        a = value(1)
        b = value(2)
        c = a + b
        nodes, edges = c._trace()
        self.assertEqual(set(nodes), {a, b, c})
        self.assertEqual(len(edges), 2)


class NetworkTests(unittest.TestCase):
    def test_parameter_count(self):
        net = network(num_in=2, layer_sizes=[3, 1], unit_activation='relu')
        self.assertEqual(len(net.layers), 2)
        self.assertEqual(len(net.parameters()), 3 * 3 + 1 * 4)

    def test_parameter_count_deep(self):
        net = network(num_in=2, layer_sizes=[3, 2, 1], unit_activation='tanh')
        self.assertEqual(len(net.parameters()), 3 * 3 + 2 * 4 + 1 * 3)

    def test_forward_returns_value(self):
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='tanh')
        out = net([1, -1])
        self.assertIsInstance(out, value)
        self.assertTrue(math.isfinite(out.data))

    def test_predict(self):
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='relu')
        predictions = net.predict([[0, 0], [1, 1]])
        self.assertEqual(len(predictions), 2)
        self.assertTrue(all(isinstance(p, float) or isinstance(p, int) for p in predictions))

    def test_zero_grad(self):
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='sigmoid')
        loss = net.mse_loss([[0, 0], [1, 1]], [[0], [1]])
        loss.backward()
        self.assertTrue(any(p.gradient != 0 for p in net.parameters()))
        net.zero_grad()
        self.assertTrue(all(p.gradient == 0 for p in net.parameters()))

    def test_neuron_relu(self):
        n = neuron(2, _activation='relu')
        n.weights[0].data = 1
        n.weights[1].data = -1
        n.bias.data = 0
        out = n([2, 3])
        self.assertEqual(out.data, 0)

    def test_neuron_parameter_count(self):
        self.assertEqual(len(neuron(4).parameters()), 5)

    def test_neuron_default_activation_is_tanh(self):
        n = neuron(2)
        n.weights[0].data = 1
        n.weights[1].data = 0
        n.bias.data = 0
        out = n([2, 5])
        self.assertAlmostEqual(out.data, math.tanh(2))

    def test_layer_single_output_returns_value(self):
        l = layer(3, 1)
        self.assertIsInstance(l([1, 2, 3]), value)

    def test_layer_multi_output_returns_list(self):
        l = layer(2, 3)
        out = l([1, 2])
        self.assertEqual(len(out), 3)
        self.assertTrue(all(isinstance(o, value) for o in out))

    def test_layer_parameter_count(self):
        self.assertEqual(len(layer(3, 4).parameters()), 4 * (3 + 1))

    def test_layer_activations(self):
        for act in ('sigmoid', 'tanh', 'relu'):
            l = layer(2, 1, act)
            out = l([0, 0])
            self.assertTrue(math.isfinite(out.data))

    def test_mse_loss_is_non_negative(self):
        random.seed(0)
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='tanh')
        loss = net.mse_loss([[0, 0], [1, 1]], [[0], [1]])
        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(loss.data, 0)

    def test_svm_loss_returns_value_and_accuracy(self):
        random.seed(1)
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='tanh')
        loss, accuracy = net.svm_loss([[1, 0], [0, 1]], [1, -1])
        self.assertIsInstance(loss, value)
        self.assertGreaterEqual(loss.data, 0)
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_fit_updates_parameters(self):
        random.seed(2)
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='tanh')
        before = [p.data for p in net.parameters()]
        with redirect_stdout(io.StringIO()):
            net.fit([[0, 0], [1, 1]], [[0], [1]], epochs=2)
        after = [p.data for p in net.parameters()]
        self.assertTrue(any(b != a for b, a in zip(before, after)))

    def test_summary_prints_parameter_count(self):
        net = network(num_in=2, layer_sizes=[2, 1], unit_activation='relu')
        buf = io.StringIO()
        with redirect_stdout(buf):
            net.summary()
        output = buf.getvalue()
        self.assertIn('Total number of parameters', output)
        self.assertIn(str(len(net.parameters())), output)


if __name__ == '__main__':
    unittest.main()
