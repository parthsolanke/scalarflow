import math

import pytest

from scalarflow.engine import value


def test_arithmetic_forward():
    a = value(2)
    b = value(3)

    assert (a + b).data == 5
    assert (a - b).data == -1
    assert (a * b).data == 6
    assert (a / b).data == pytest.approx(2 / 3)
    assert (a ** 2).data == 4
    assert (2 + a).data == 4
    assert (2 - a).data == 0
    assert (2 * a).data == 4
    assert (6 / a).data == 3
    assert (-a).data == -2


def test_backward_on_add_and_mul():
    a = value(2)
    b = value(3)
    z = a * b + a

    z.backward()

    assert z.gradient == 1
    assert a.gradient == pytest.approx(4)
    assert b.gradient == pytest.approx(2)


def test_backward_on_power_and_division():
    a = value(2)
    z = a ** 3

    z.backward()

    assert z.data == 8
    assert a.gradient == pytest.approx(12)


def test_activation_forward_and_gradients():
    x = value(0.5)

    t = x.tanh()
    t.backward()
    assert t.data == pytest.approx(math.tanh(0.5))
    assert x.gradient == pytest.approx(1 - math.tanh(0.5) ** 2)

    y = value(0.25)
    s = y.sigmoid()
    s.backward()
    expected = 1 / (1 + math.exp(-0.25))
    assert s.data == pytest.approx(expected)
    assert y.gradient == pytest.approx(expected * (1 - expected))

    r = value(-1.5).relu()
    r.backward()
    assert r.data == 0
    assert r.gradient == 0

    p = value(1.5).relu()
    p.backward()
    assert p.data == 1.5
    assert p.gradient == 1


def test_exp_forward_and_gradient():
    x = value(0.3)
    e = x.exp()

    e.backward()

    assert e.data == pytest.approx(math.exp(0.3))
    assert x.gradient == pytest.approx(math.exp(0.3))


def test_repr_shows_data_and_gradient():
    assert "data" in repr(value(1.0))
    assert "grad" in repr(value(1.0))
