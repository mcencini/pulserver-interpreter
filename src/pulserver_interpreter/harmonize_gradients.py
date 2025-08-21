"""Harmonize gradients within a block to the same (potentially non-uniform) time raster"""

__all__ = ["harmonize_gradients"]

from types import SimpleNamespace

import numpy as np

from pypulseq import eps
from pypulseq.opts import Opts
from pypulseq.calc_duration import calc_duration
from pypulseq.make_arbitrary_grad import make_arbitrary_grad
from pypulseq.make_extended_trapezoid import make_extended_trapezoid
from pypulseq.make_trapezoid import make_trapezoid
from pypulseq.points_to_waveform import points_to_waveform
from pypulseq.utils.cumsum import cumsum


def harmonize_gradients(block: SimpleNamespace, system: Opts) -> SimpleNamespace:
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
    system : Opts
        System limits.

    Returns
    -------
    block : SimpleNamespace
        The same block with gx, gy, gz replaced by harmonized gradient events.
    """
    grad_attrs = ["gx", "gy", "gz"]

    # Collect present gradients and their meta
    grads = {}
    has_grad = []
    for attr in grad_attrs:
        grad = getattr(block, attr, None)
        if grad is not None:
            grads[attr] = grad
        has_grad.append(grad is not None)

    if not any(has_grad):
        return block

    # Check if we have a set of traps with the same timing
    g0 = next(iter(grads.values()))
    if (
        np.all([g.type == "trap" for g in list(grads.values())])
        and np.all([g.rise_time == g0.rise_time for g in list(grads.values())])
        and np.all([g.flat_time == g0.flat_time for g in list(grads.values())])
        and np.all([g.fall_time == g0.fall_time for g in list(grads.values())])
        and np.all([g.delay == g0.delay for g in list(grads.values())])
    ):
        return _prep_output(block, grads)

    # Find out the general delay of all gradients and other statistics
    delays, firsts, lasts, durs, is_trap, is_arb, is_osa = {}, {}, {}, {}, {}, {}, {}
    for channel, g in grads.items():
        delays[channel] = g.delay
        durs[channel] = calc_duration(g)
        is_trap[channel] = g.type == "trap"
        if is_trap[channel]:
            is_arb[channel] = False
            is_osa[channel] = False
            firsts[channel] = 0.0
            lasts[channel] = 0.0
        else:
            tt_rast = g.tt / system.grad_raster_time
            is_arb[channel] = (
                np.all(np.abs(tt_rast + 0.5 - np.arange(1, len(tt_rast) + 1))) < eps
            )
            is_osa[channel] = np.all(
                np.abs(tt_rast - 0.5 * np.arange(1, len(tt_rast) + 1)) < eps
            )
            firsts[channel] = g.first
            lasts[channel] = g.last

    # Check if we only have arbitrary grads on irregular time samplings, optionally mixed with trapezoids
    is_etrap = np.logical_and.reduce(
        (
            np.logical_not(list(is_trap.values())),
            np.logical_not(list(is_arb.values())),
            np.logical_not(list(is_osa.values())),
        )
    )
    if np.all(np.logical_or(list(is_trap.values()), is_etrap)):
        # Keep shapes still rather simple
        times = []
        for g in grads.values():
            if g.type == "trap":
                times.extend(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
            else:
                times.extend(g.delay + g.tt)

        times = np.unique(times)
        dt = times[1:] - times[:-1]
        ieps = np.flatnonzero(dt < eps)
        if np.any(ieps):
            dtx = np.array([times[0], *dt])
            dtx[ieps] = (
                dtx[ieps] + dtx[ieps + 1]
            )  # Assumes that no more than two too similar values can occur
            dtx = np.delete(dtx, ieps + 1)
            times = np.cumsum(dtx)

        for channel, g in grads.items():
            if g.type == "trap":
                if g.flat_time > 0:  # Trapezoid or triangle
                    tt = list(cumsum(g.delay, g.rise_time, g.flat_time, g.fall_time))
                    waveform = [0, g.amplitude, g.amplitude, 0]
                else:
                    tt = list(cumsum(g.delay, g.rise_time, g.fall_time))
                    waveform = [0, g.amplitude, 0]
            else:
                tt = g.delay + g.tt
                waveform = g.waveform

            # Fix rounding for the first and last time points
            i_min = np.argmin(np.abs(tt[0] - times))
            t_min = (np.abs(tt[0] - times))[i_min]
            if t_min < eps:
                tt[0] = times[i_min]
            i_min = np.argmin(np.abs(tt[-1] - times))
            t_min = (np.abs(tt[-1] - times))[i_min]
            if t_min < eps:
                tt[-1] = times[i_min]

            if np.abs(waveform[0]) > eps and tt[0] > eps:
                tt[0] += eps

            amplitudes = np.interp(xp=tt, fp=waveform, x=times, left=0, right=0)
            grads[channel] = make_extended_trapezoid(
                channel=g.channel, amplitudes=amplitudes, times=times, system=system
            )

        return _prep_output(block, grads)

    # Convert to numpy.ndarray for fancy-indexing later on
    firsts, lasts = np.array(list(firsts.values())), np.array(list(lasts.values()))
    common_delay = np.min(list(delays.values()))
    durs = np.array(list(durs.values()))
    total_duration = np.max(durs)

    # Convert everything to a regularly-sampled waveform
    waveforms = {}
    max_length = 0

    if np.any(list(is_osa.values())):
        target_raster = 0.5 * system.grad_raster_time
    else:
        target_raster = system.grad_raster_time

    for channel, g in grads.items():
        if g.type == "grad":
            if is_arb[channel] or is_osa[channel]:
                if (
                    np.any(list(is_osa.values())) and is_arb[channel]
                ):  # Porting MATLAB here, maybe a bit ugly
                    # Interpolate missing samples
                    idx = np.arange(0, len(g.waveform) - 0.5 + eps, 0.5)
                    wf = g.waveform
                    interp_waveform = 0.5 * (
                        wf[np.floor(idx).astype(int)] + wf[np.ceil(idx).astype(int)]
                    )
                    waveforms[channel] = interp_waveform
                else:
                    waveforms[channel] = g.waveform
            else:
                waveforms[channel] = points_to_waveform(
                    amplitudes=g.waveform,
                    times=g.tt,
                    grad_raster_time=target_raster,
                )
        elif g.type == "trap":
            if g.flat_time > 0:  # Triangle or trapezoid
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.flat_time,
                        g.delay
                        - common_delay
                        + g.rise_time
                        + g.flat_time
                        + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, g.amplitude, 0])
            else:
                times = np.array(
                    [
                        g.delay - common_delay,
                        g.delay - common_delay + g.rise_time,
                        g.delay - common_delay + g.rise_time + g.fall_time,
                    ]
                )
                amplitudes = np.array([0, g.amplitude, 0])
            waveforms[channel] = points_to_waveform(
                amplitudes=amplitudes,
                times=times,
                grad_raster_time=target_raster,
            )
        else:
            raise ValueError("Unknown gradient type")

        if g.delay - common_delay > 0:
            # Stop for numpy.arange is not g.delay - common_delay - system.grad_raster_time like in Matlab
            # so as to include the endpoint
            waveforms[channel] = np.concatenate(
                (
                    np.zeros(round((g.delay - common_delay) / system.grad_raster_time)),
                    waveforms[channel],
                )
            )

        num_points = len(waveforms[channel])
        max_length = max(num_points, max_length)

    # Pad all waveforms to max length
    for channel, g in grads.items():
        wt = np.zeros(max_length)
        wt[0 : len(waveforms[channel])] = waveforms[channel]

        grads[channel] = make_arbitrary_grad(
            channel=g.channel,
            waveform=wt,
            system=system,
            max_slew=system.max_slew,
            max_grad=system.max_grad,
            delay=common_delay,
            oversampling=np.any(list(is_osa.values())),
            first=np.sum(firsts[delays == common_delay]),
            last=np.sum(lasts[durs == total_duration]),
        )

        # Fix the first and the last values
        # First is defined by the sum of firsts with the minimal delay (common_delay)
        # Last is defined by the sum of lasts with the maximum duration (total_duration == durs.max())
        grads[channel].first = np.sum(firsts[np.array(delays) == common_delay])
        grads[channel].last = np.sum(lasts[durs == total_duration])

    return _prep_output(block, grads)


def _prep_output(block, grads):
    channels = ["x", "y", "z"]
    grad_attrs = ["gx", "gy", "gz"]

    # get type and raster of harmonized gradient
    g0 = next(iter(grads.values()))
    gtype = g0.type
    if gtype == "trap":
        times = (g0.delay, g0.rise_time, g0.flat_time, g0.fall_time)
        amplitude = 0
    else:
        delay = g0.delay
        times = g0.tt
        amplitudes = np.zeros_like(g0.tt)

    for n in range(3):
        attr = grad_attrs[n]
        ch = channels[n]
        if attr in grads:
            setattr(block, attr, grads[attr])
        else:
            if gtype == "trap":
                setattr(
                    block,
                    attr,
                    make_trapezoid(
                        channel=ch,
                        amplitude=amplitude,
                        delay=times[0],
                        rise_time=times[1],
                        flat_time=times[2],
                        fall_time=times[3],
                    ),
                )
            else:
                grad = SimpleNamespace()
                grad.type = "grad"
                grad.channel = ch
                grad.waveform = amplitudes
                grad.delay = delay
                grad.tt = times
                grad.shape_dur = grad.tt[-1]
                grad.area = 0.0
                grad.first = 0.0
                grad.last = 0.0
                setattr(block, attr, grad)

    return block
