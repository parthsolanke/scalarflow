import math
import unittest

from scalarflow.engine import value


class ArithmeticTests(unittest.TestCase):

    def test_addition(self):
        self.assertEqual((value(2) + value(3)).data, 5)
        self.assertEqual((value(2) + 3).data, 5)
        self.assertEqual((2 + value(3)).data, 5)

    def test_subtraction(self):
        self.assertEqual((value(5) - value(2)).data, 3)
        self.assertEqual((value(5) - 2).data, 3)
        self.assertEqual((5 - value(2)).data, 3)

    def test_multiplication(self):
        self.assertEqual((value(4) * value(3)).data, 12)
        self.assertEqual((value(4) * 3).data, 12)
        self.assertEqual((3 * value(4)).data, 12)

    def test_division(self):
        self.assertEqual((value(6) / value(3)).data, 2)
        self.assertEqual((value(6) / 3).data, 2)
        self.assertEqual((6 / value(3)).data, 2)

    def test_power(self):
        self.assertEqual((value(2) ** 3).data, 8)

    def test_negation(self):
        self.assertEqual((-value(3)).data, -3)


class BackwardTests(unittest.TestCase):

    def test_addition_gradients(self):
        a, b = value(2), value(3)
        c = a + b
        c.backward()

        self.assertEqual(c.gradient, 1)
        self.assertEqual(a.gradient, 1)
        self.assertEqual(b.gradient, 1)

    def test_multiplication_gradients(self):
        a, b = value(2), value(3)
        c = a * b
        c.backward()

        self.assertEqual(c.gradient, 1)
        self.assertEqual(a.gradient, 3)
        self.assertEqual(b.gradient, 2)

    def test_reused_node_gradients(self):
        x = value(2)
        y = x * x + x
        y.backward()

        self.assertAlmostEqual(x.gradient, 5)

    def test_power_gradient(self):
        x = value(3)
        y = x ** 2
        y.backward()

        self.assertAlmostEqual(x.gradient, 6)

    def test_division_gradients(self):
        x, y = value(2), value(4)
        z = x / y
        z.backward()

        self.assertAlmostEqual(x.gradient, 0.25)
        self.assertAlmostEqual(y.gradient, -0.125)

    def test_exp_gradient(self):
        x = value(0)
        y = x.exp()
        y.backward()

        self.assertAlmostEqual(y.data, 1)
        self.assertAlmostEqual(x.gradient, 1)

    def test_chain_rule(self):
        x = value(1)
        y = (x * 2 + 1) ** 2
        y.backward()

        self.assertAlmostEqual(y.data, 9)
        self.assertAlmostEqual(x.gradient, 12)


class ActivationTests(unittest.TestCase):

    def test_tanh_forward(self):
        self.assertAlmostEqual(value(1).tanh().data, math.tanh(1))

    def test_tanh_gradient(self):
        x = value(0)
        y = x.tanh()
        y.backward()

        self.assertAlmostEqual(x.gradient, 1)

    def test_relu_forward(self):
        self.assertEqual(value(-2).relu().data, 0)
        self.assertEqual(value(3).relu().data, 3)

    def test_relu_gradient(self):
        negative = value(-2)
        negative.relu().backward()
        self.assertEqual(negative.gradient, 0)

        positive = value(3)
        positive.relu().backward()
        self.assertEqual(positive.gradient, 1)

    def test_sigmoid_forward(self):
        self.assertAlmostEqual(value(0).sigmoid().data, 0.5)
        self.assertAlmostEqual(value(1).sigmoid().data, 1 / (1 + math.exp(-1)))

    def test_sigmoid_gradient(self):
        x = value(0)
        y = x.sigmoid()
        y.backward()

        self.assertAlmostEqual(x.gradient, 0.25)


class RepresentationTests(unittest.TestCase):

    def test_repr(self):
        self.assertEqual(repr(value(5)), 'value(data = 5, grad = 0)')


if __name__ == '__main__':
    unittest.main()
