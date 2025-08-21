"""Test suite for harmonize_gradients."""

import numpy as np
from types import SimpleNamespace

from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.opts import Opts
from pypulseq.calc_duration import calc_duration

from pulserver_interpreter import harmonize_gradients

# --- Helper functions using pypulseq ---

sys = Opts(max_grad=np.inf, max_slew=np.inf, grad_raster_time=0.5e-3)


def make_trap(channel="x", amplitude=1.0, rise=1e-3, flat=2e-3, fall=1e-3, delay=0.0):
    # Use default Opts, no system limits
    return make_trapezoid(
        channel=channel,
        amplitude=amplitude,
        rise_time=rise,
        flat_time=flat,
        fall_time=fall,
        delay=delay,
        system=sys,
    )


def make_ext_trap(channel="x", times=None, amps=None):
    # Extended trapezoid: non-uniform times
    if times is None:
        times = np.array([0.0, 1e-3, 3e-3])
    if amps is None:
        amps = np.array([0.0, 1.0, 0.0])
    return make_extended_trapezoid(
        channel=channel,
        amplitudes=amps,
        times=times,
        system=sys,
    )


def make_arb(channel="x", waveform=None, delay=0.0):
    if waveform is None:
        waveform = np.array([0.0, 1.0, 0.0])
    return make_arbitrary_grad(
        channel=channel,
        waveform=waveform,
        delay=delay,
        system=sys,
    )


def check_block_waveforms(block):
    """Check that gx, gy, gz exist and waveforms are harmonized in time."""
    times = []
    for ch in ["gx", "gy", "gz"]:
        grad = getattr(block, ch, None)
        assert grad is not None, f"{ch} missing from block"
        if grad.type == "trap":
            t = (
                np.array(
                    [
                        grad.delay,
                        grad.delay + grad.rise_time,
                        grad.delay + grad.rise_time + grad.flat_time,
                        grad.delay + grad.rise_time + grad.flat_time + grad.fall_time,
                    ]
                )
                if grad.flat_time > 0
                else np.array(
                    [
                        grad.delay,
                        grad.delay + grad.rise_time,
                        grad.delay + grad.rise_time + grad.fall_time,
                    ]
                )
            )
            times.append(np.linspace(0, calc_duration(grad), len(t)))
        elif grad.type == "grad":
            times.append(grad.delay + grad.tt)
        else:
            raise AssertionError(f"Unknown grad type {grad.type} for channel {ch}")
    # All three time arrays (from gx, gy, gz) should match in shape and values
    for t in times[1:]:
        assert np.allclose(
            t, times[0]
        ), f"Time arrays of gx, gy, gz are not equal: {t} vs {times[0]}"


# --- Test cases ---


def test_case_1_no_gradients():
    block = SimpleNamespace()
    new_block = harmonize_gradients(block, sys)
    # Should exit without adding any gradients
    assert not hasattr(new_block, "gx")
    assert not hasattr(new_block, "gy")
    assert not hasattr(new_block, "gz")


def test_case_2_single_trap():
    block = SimpleNamespace(gx=make_trap(channel="x"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_3_single_ext_trap():
    block = SimpleNamespace(gx=make_ext_trap(channel="x"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_4_single_arb():
    block = SimpleNamespace(gx=make_arb(channel="x"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_5_two_trap_same_timing():
    block = SimpleNamespace(
        gx=make_trap(channel="x"), gy=make_trap(channel="y", amplitude=2.0)
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_6_two_trap_different_timing():
    block = SimpleNamespace(
        gx=make_trap(channel="x", flat=1e-3), gy=make_trap(channel="y", flat=2e-3)
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_7_two_ext_trap_same_raster():
    block = SimpleNamespace(
        gx=make_ext_trap(channel="x"),
        gy=make_ext_trap(channel="y", amps=np.array([0.0, 0.5, 0.0])),
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_8_two_ext_trap_different_raster():
    block = SimpleNamespace(
        gx=make_ext_trap(channel="x"),
        gy=make_ext_trap(channel="y", times=np.array([0.0, 0.5e-3, 2e-3])),
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_9_two_arb_raster():
    block = SimpleNamespace(
        gx=make_arb(channel="x"),
        gy=make_arb(channel="y", waveform=np.array([0, 0.5, 0])),
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_10_trap_plus_ext_trap():
    block = SimpleNamespace(gx=make_trap(channel="x"), gy=make_ext_trap(channel="y"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_11_trap_plus_arb():
    block = SimpleNamespace(gx=make_trap(channel="x"), gy=make_arb(channel="y"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_12_ext_trap_plus_arb():
    block = SimpleNamespace(gx=make_ext_trap(channel="x"), gy=make_arb(channel="y"))
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)


def test_case_13_trap_ext_trap_arb():
    block = SimpleNamespace(
        gx=make_trap(channel="x"),
        gy=make_ext_trap(channel="y"),
        gz=make_arb(channel="z"),
    )
    new_block = harmonize_gradients(block, sys)
    check_block_waveforms(new_block)
