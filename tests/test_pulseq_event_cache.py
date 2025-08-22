"""PyPulseq replacement routine."""

from types import SimpleNamespace

import numpy as np

from pulserver_interpreter.pulseq import PyPulseq

import pypulseq

pp = PyPulseq(pypulseq)


# ---------- Basic caching ----------
def test_basic_cache():
    rf1 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=500e-6)
    rf2 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=500e-6)
    # Should be cached (deepcopy)
    assert rf1 is not rf2
    assert np.allclose(rf1.signal, rf2.signal)


# ---------- Different args triggers recompute ----------
def test_cache_diff_args():
    rf1 = pp.make_block_pulse(flip_angle=np.pi / 2, duration=500e-6)
    rf2 = pp.make_block_pulse(flip_angle=np.pi / 4, duration=500e-6)
    # Different flip angle → recompute
    assert not np.allclose(rf1.signal, rf2.signal)


# ---------- Custom user-defined function ----------
def test_custom_event_cache():
    @pp.cached_event
    def my_event(val):
        ns = SimpleNamespace()
        ns.val = val
        ns.type = "custom"
        return ns

    e1 = my_event(10)
    e2 = my_event(10)
    # Cached deepcopy returned
    assert e1 is not e2
    assert e1.val == e2.val

    e3 = my_event(20)
    # Different argument → recompute
    assert e3.val != e1.val


# ---------- Multi-event output ----------
def test_multi_event_output():
    @pp.cached_event
    def multi_event(a):
        return (
            pp.make_block_pulse(flip_angle=a * np.pi / 180, duration=500e-6),
            pp.make_trapezoid("x", area=1000.0),
        )

    e1 = multi_event(45)
    e2 = multi_event(45)
    # Should be cached deepcopy of tuple
    assert e1 is not e2
    assert np.allclose(e1[0].signal, e2[0].signal)
    assert e1[1].amplitude == e2[1].amplitude
