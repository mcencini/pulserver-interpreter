"""Harmonize gradients within a block to the same (potentially non-uniform) time raster"""

__all__ = ["harmonize_gradients"]

from types import SimpleNamespace

import numpy as np

from pypulseq import eps
from pypulseq.opts import Opts
from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.utils.tracing import trace, trace_enabled


def harmonize_gradients(block: SimpleNamespace) -> SimpleNamespace:
    """
    Harmonize the gradients in a Pypulseq block to a common raster time.

    This function takes a Pypulseq block with at most one gradient event on each
    of the 'x', 'y', and 'z' channels (as attributes gx, gy, gz), and adapts all non-empty
    gradients to a common raster time, producing harmonized gradients. The harmonized
    gradients are set back as gx, gy, gz attributes of the input block, replacing the
    originals. Empty channels are filled with a harmonized zero-amplitude (or zero-waveform)
    gradient event. If there are no gradient events, the input block is returned as is.

    Parameters
    ----------
    block : SimpleNamespace
        A Pypulseq block, possibly containing gx, gy, gz attributes (gradient events)
        on 'x', 'y', and 'z' channels, respectively.

    Returns
    -------
    block : SimpleNamespace
        The same block with gx, gy, gz replaced by harmonized gradient events.
    """
    channels = ["x", "y", "z"]
    grad_attrs = ["gx", "gy", "gz"]

    # Collect present gradients and their meta
    grads = []
    has_grad = []
    for attr in grad_attrs:
        grad = getattr(block, attr, None)
        grads.append(grad)
        has_grad.append(grad is not None)

    if not any(has_grad):
        return block

    # Use dummy system to skip limits
    system = Opts(max_grad=np.inf, max_slew=np.inf)

    # Gather delays, durations, and distinguish types
    delays, durs, is_trap, is_arb, is_osa = [], [], [], [], []
    firsts, lasts = [], []
    for grad in grads:
        if grad is None:
            delays.append(0)
            durs.append(0)
            is_trap.append(False)
            is_arb.append(False)
            is_osa.append(False)
            firsts.append(0.0)
            lasts.append(0.0)
            continue

        delays.append(grad.delay)
        durs.append(calc_duration(grad))
        is_trap.append(grad.type == "trap")
        if grad.type == "trap":
            is_arb.append(False)
            is_osa.append(False)
            firsts.append(0.0)
            lasts.append(0.0)
        else:  # grad.type == 'grad'
            tt = grad.tt
            # Check if tt is uniform (arbitrary)
            rast = system.grad_raster_time
            is_uniform = np.all(
                np.abs(tt / rast + 0.5 - np.arange(1, len(tt) + 1)) < eps
            )
            is_arb.append(is_uniform)
            # Check for oversampling (OSA)
            is_osa.append(
                np.all(
                    np.abs(tt / (0.5 * rast) - 0.5 * np.arange(1, len(tt) + 1)) < eps
                )
            )
            firsts.append(getattr(grad, "first", 0.0))
            lasts.append(getattr(grad, "last", 0.0))

    # Find the minimal delay and maximal duration
    common_delay = min(
        [d for (g, d) in zip(has_grad, delays, strict=False) if g] or [0.0]
    )
    total_duration = max(
        [d for (g, d) in zip(has_grad, durs, strict=False) if g] or [0.0]
    )

    # Pick the target raster time
    target_raster = (
        (0.5 * system.grad_raster_time) if any(is_osa) else system.grad_raster_time
    )

    # Convert each gradient to a waveform on the common raster
    harmonized = []
    max_length = (
        int(np.ceil((total_duration) / target_raster)) if total_duration > 0 else 1
    )

    for idx, (grad, channel) in enumerate(zip(grads, channels, strict=False)):
        if grad is None:
            waveform = np.zeros(max_length, dtype=float)
        elif grad.type == "trap":
            # Trapezoid or triangle
            if grad.flat_time > 0:
                times = np.array(
                    [
                        grad.delay - common_delay,
                        grad.delay - common_delay + grad.rise_time,
                        grad.delay - common_delay + grad.rise_time + grad.flat_time,
                        grad.delay
                        - common_delay
                        + grad.rise_time
                        + grad.flat_time
                        + grad.fall_time,
                    ]
                )
                amplitudes = np.array([0, grad.amplitude, grad.amplitude, 0])
            else:
                times = np.array(
                    [
                        grad.delay - common_delay,
                        grad.delay - common_delay + grad.rise_time,
                        grad.delay - common_delay + grad.rise_time + grad.fall_time,
                    ]
                )
                amplitudes = np.array([0, grad.amplitude, 0])
            waveform = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=target_raster,
            )
        elif grad.type == "grad":
            tt = grad.tt
            # For OSA/arb/extended we always interpolate onto the common raster
            times = grad.delay - common_delay + tt
            waveform = points_to_waveform(
                amplitudes=grad.waveform,
                times=times,
                grad_raster_time=target_raster,
            )
        else:
            raise ValueError(f"Unknown gradient type: {grad.type}")

        # Pad to common length
        if len(waveform) < max_length:
            waveform = np.pad(waveform, (0, max_length - len(waveform)))
        if len(waveform) > max_length:
            waveform = waveform[:max_length]

        harm_grad = make_arbitrary_grad(
            channel=channel,
            waveform=waveform,
            system=system,
            delay=common_delay,
            max_slew=system.max_slew,
            max_grad=system.max_grad,
            oversampling=any(is_osa),
            first=firsts[idx] if delays[idx] == common_delay else 0.0,
            last=lasts[idx] if durs[idx] == total_duration else 0.0,
        )
        if trace_enabled():
            harm_grad.trace = trace()
        harmonized.append(harm_grad)

    # Assign back
    for attr, grad in zip(grad_attrs, harmonized, strict=False):
        setattr(block, attr, grad)
    return block
