import math
import unittest

from scalarflow.engine import value


class TestValueOperations(unittest.TestCase):

    def test_add(self):
        a = value(2)
        b = value(3)

        c = a + b
        c.backward()

        self.assertEqual(c.data, 5)
        self.assertAlmostEqual(a.gradient, 1)
        self.assertAlmostEqual(b.gradient, 1)

    def test_radd(self):
        a = value(2)

        c = 3 + a
        c.backward()

        self.assertEqual(c.data, 5)
        self.assertAlmostEqual(a.gradient, 1)

    def test_neg_and_sub(self):
        a = value(3)
        b = value(1)

        c = a - b
        c.backward()

        self.assertEqual(c.data, 2)
        self.assertAlmostEqual(a.gradient, 1)
        self.assertAlmostEqual(b.gradient, -1)

    def test_rsub(self):
        a = value(3)

        c = 10 - a
        c.backward()

        self.assertEqual(c.data, 7)
        self.assertAlmostEqual(a.gradient, -1)

    def test_mul(self):
        a = value(2)
        b = value(3)

        c = a * b
        c.backward()

        self.assertEqual(c.data, 6)
        self.assertAlmostEqual(a.gradient, 3)
        self.assertAlmostEqual(b.gradient, 2)

    def test_rmul(self):
        a = value(4)

        c = 5 * a
        c.backward()

        self.assertEqual(c.data, 20)
        self.assertAlmostEqual(a.gradient, 5)

    def test_pow(self):
        a = value(3)

        c = a ** 2
        c.backward()

        self.assertEqual(c.data, 9)
        self.assertAlmostEqual(a.gradient, 6)

    def test_truediv(self):
        a = value(6)
        b = value(3)

        c = a / b
        c.backward()

        self.assertEqual(c.data, 2)
        self.assertAlmostEqual(a.gradient, 1 / 3)
        self.assertAlmostEqual(b.gradient, -2 / 3)

    def test_rtruediv(self):
        a = value(4)

        c = 8 / a
        c.backward()

        self.assertEqual(c.data, 2)
        self.assertAlmostEqual(a.gradient, -8 / 16)

    def test_repr(self):
        a = value(2)
        a.gradient = 1

        self.assertEqual(repr(a), 'value(data = 2, grad = 1)')


class TestValueActivations(unittest.TestCase):

    def test_exp(self):
        a = value(0)

        c = a.exp()
        c.backward()

        self.assertAlmostEqual(c.data, 1)
        self.assertAlmostEqual(a.gradient, 1)

    def test_tanh(self):
        a = value(0)

        c = a.tanh()
        c.backward()

        self.assertAlmostEqual(c.data, 0)
        self.assertAlmostEqual(a.gradient, 1)

    def test_tanh_matches_math(self):
        a = value(1.0)

        c = a.tanh()

        self.assertAlmostEqual(c.data, math.tanh(1.0))

    def test_relu_positive(self):
        a = value(2)

        c = a.relu()
        c.backward()

        self.assertEqual(c.data, 2)
        self.assertAlmostEqual(a.gradient, 1)

    def test_relu_negative(self):
        a = value(-2)

        c = a.relu()
        c.backward()

        self.assertEqual(c.data, 0)
        self.assertAlmostEqual(a.gradient, 0)

    def test_sigmoid(self):
        a = value(0)

        c = a.sigmoid()
        c.backward()

        self.assertAlmostEqual(c.data, 0.5)
        self.assertAlmostEqual(a.gradient, 0.25)


class TestBackward(unittest.TestCase):

    def test_chain_rule(self):
        a = value(2)
        b = value(3)

        c = a * b
        d = c + a
        d.backward()

        self.assertEqual(d.data, 8)
        self.assertAlmostEqual(a.gradient, 4)
        self.assertAlmostEqual(b.gradient, 2)

    def test_composite_expression(self):
        a = value(2)
        b = value(-3)
        c = value(10)

        e = a * b + c
        e.backward()

        self.assertEqual(e.data, 4)
        self.assertAlmostEqual(a.gradient, -3)
        self.assertAlmostEqual(b.gradient, 2)
        self.assertAlmostEqual(c.gradient, 1)


if __name__ == '__main__':
    unittest.main()
