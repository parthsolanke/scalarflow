import math
import unittest

from scalarflow.engine import value


class ValueInterfaceTests(unittest.TestCase):
    def test_repr(self):
        self.assertEqual(repr(value(3)), "value(data = 3, grad = 0)")
        self.assertEqual(repr(value(1.5)), "value(data = 1.5, grad = 0)")

    def test_defaults(self):
        v = value(2, label="a")
        self.assertEqual(v.data, 2)
        self.assertEqual(v.gradient, 0)
        self.assertEqual(v.label, "a")
        self.assertEqual(v._operation, "")
        self.assertEqual(v._prev, set())


class ArithmeticTests(unittest.TestCase):
    def test_addition(self):
        a, b = value(2), value(3)
        c = a + b
        self.assertEqual(c.data, 5)
        c.backward()
        self.assertEqual(a.gradient, 1)
        self.assertEqual(b.gradient, 1)

    def test_reverse_addition(self):
        a = value(4)
        c = 3 + a
        self.assertEqual(c.data, 7)
        c.backward()
        self.assertEqual(a.gradient, 1)

    def test_negation(self):
        a = value(5)
        c = -a
        self.assertEqual(c.data, -5)
        c.backward()
        self.assertEqual(a.gradient, -1)

    def test_subtraction(self):
        a, b = value(5), value(2)
        c = a - b
        self.assertEqual(c.data, 3)
        c.backward()
        self.assertEqual(a.gradient, 1)
        self.assertEqual(b.gradient, -1)

    def test_reverse_subtraction(self):
        a = value(5)
        c = 2 - a
        self.assertEqual(c.data, -3)
        c.backward()
        self.assertEqual(a.gradient, -1)

    def test_multiplication(self):
        a, b = value(2), value(3)
        c = a * b
        self.assertEqual(c.data, 6)
        c.backward()
        self.assertEqual(a.gradient, 3)
        self.assertEqual(b.gradient, 2)

    def test_reverse_multiplication(self):
        a = value(4)
        c = 3 * a
        self.assertEqual(c.data, 12)
        c.backward()
        self.assertEqual(a.gradient, 3)

    def test_division(self):
        a, b = value(6), value(3)
        c = a / b
        self.assertAlmostEqual(c.data, 2)
        c.backward()
        self.assertAlmostEqual(a.gradient, 1 / 3)
        self.assertAlmostEqual(b.gradient, -6 / 9)

    def test_reverse_division(self):
        a = value(2)
        c = 3 / a
        self.assertAlmostEqual(c.data, 1.5)
        c.backward()
        self.assertAlmostEqual(a.gradient, -3 / 4)

    def test_power(self):
        a = value(3)
        c = a ** 2
        self.assertEqual(c.data, 9)
        c.backward()
        self.assertAlmostEqual(a.gradient, 6)

    def test_power_rejects_non_number(self):
        with self.assertRaises(AssertionError):
            value(2) ** value(3)


class ActivationTests(unittest.TestCase):
    def test_exp(self):
        a = value(1)
        c = a.exp()
        self.assertAlmostEqual(c.data, math.e)
        c.backward()
        self.assertAlmostEqual(a.gradient, math.e)

    def test_tanh(self):
        a = value(0.5)
        c = a.tanh()
        expected = (math.exp(1) - 1) / (math.exp(1) + 1)
        self.assertAlmostEqual(c.data, expected)
        c.backward()
        self.assertAlmostEqual(a.gradient, 1 - c.data ** 2)

    def test_relu_positive(self):
        a = value(2)
        c = a.relu()
        self.assertEqual(c.data, 2)
        c.backward()
        self.assertEqual(a.gradient, 1)

    def test_relu_negative(self):
        a = value(-2)
        c = a.relu()
        self.assertEqual(c.data, 0)
        c.backward()
        self.assertEqual(a.gradient, 0)

    def test_sigmoid(self):
        a = value(0.3)
        c = a.sigmoid()
        self.assertAlmostEqual(c.data, 1 / (1 + math.exp(-0.3)))
        c.backward()
        self.assertAlmostEqual(a.gradient, c.data * (1 - c.data))


class BackpropagationTests(unittest.TestCase):
    def test_chain_rule(self):
        a, b, c = value(2.0), value(-3.0), value(10.0)
        e = a * b
        d = e + c
        f = value(-2.0)
        loss = d * f

        loss.backward()

        self.assertEqual(loss.gradient, 1)
        self.assertAlmostEqual(a.gradient, b.data * f.data)
        self.assertAlmostEqual(b.gradient, a.data * f.data)
        self.assertAlmostEqual(c.gradient, f.data)
        self.assertAlmostEqual(f.gradient, d.data)
        self.assertAlmostEqual(e.gradient, f.data)

    def test_shared_node_accumulates_gradients(self):
        a = value(2)
        b = a + a
        b.backward()
        self.assertEqual(a.gradient, 2)

    def test_matches_numerical_gradient(self):
        def loss_fn(x):
            return ((x * x) + (2 * x) + 1).tanh()

        x = value(0.7)
        loss_fn(x).backward()

        eps = 1e-6
        numerical = (loss_fn(value(0.7 + eps)).data - loss_fn(value(0.7 - eps)).data) / (2 * eps)
        self.assertAlmostEqual(x.gradient, numerical, places=5)


if __name__ == "__main__":
    unittest.main()
