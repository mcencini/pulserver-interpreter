"""Safety evaluation utils."""

# __all__ = ["rf_params"]

import copy
import math

import numpy as np
from types import SimpleNamespace

import pypulseq as pp

from pypulseq import eps, Opts


def rf_params(rf: SimpleNamespace, system: Opts = None) -> SimpleNamespace:
    """
    Compute conservative single-channel equivalent RF parameters for RF waveform.

    Parameters
    ----------
    rf : np.ndarray
        Complex array of shape (nch, n) representing RF waveform samples.
    t : np.ndarray
        Array of time points for each rf sample.

    """
    standard_duration = 1e-3
    threshold = 0.05

    gamma = Opts().gamma if system is None else system.gamma

    # get waveform and time
    waveform = rf.signal
    t = rf.t
    dt = np.diff(t, append=t[-1])

    # Expand to (nchannel, nsamples)
    waveform = np.atleast_2d(waveform)
    nch, n = waveform.shape

    # Get complex sum and magnitude sum across channels
    waveform_cplx_sum = waveform.sum(axis=0)
    waveform_abs_sum = np.abs(waveform).sum(axis=0)

    # Get duration
    duration = dt.sum()

    # Check if static pTx
    waveform_peak = waveform_abs_sum.max()
    waveform_cplx_sum_norm = waveform_cplx_sum / (waveform_peak + eps)
    waveform_abs_sum_norm = waveform_abs_sum / (waveform_peak + eps)

    # Area
    area = np.sum(waveform_cplx_sum_norm * dt) / duration

    # Flip angle
    flip_angle = 2 * math.pi * waveform_peak * area * duration
    flip_angle = np.rad2deg(flip_angle)

    # Absolute width
    abswidth = np.sum(waveform_abs_sum_norm * dt) / duration

    # Effective width
    effwidth = np.sum(waveform_abs_sum_norm**2 * dt) / duration

    # Duty cycle
    mask = waveform_abs_sum_norm > threshold
    dtycyc = np.sum(mask * waveform_abs_sum_norm * dt) / duration

    # Max B1
    max_b1 = 1e6 * flip_angle / 360.0 / (gamma * area * duration)  # uT

    # Max integrated B1^2
    max_int_b1_sqr = max_b1**2 * effwidth * (duration / standard_duration)  # uT**2

    # Max RMS B1
    max_rms_b1 = np.sqrt(max_int_b1_sqr * (standard_duration / duration))  # uT

    # Replace effective waveform for bandwidth and isodelay calc
    rf_cplx_sum = copy.deepcopy(rf)
    rf_cplx_sum.signal = waveform_cplx_sum

    # Get bandwidth
    nom_bw = pp.calc_rf_bandwidth(rf_cplx_sum)

    # Get isodelay
    isodelay, _ = pp.calc_rf_center(rf_cplx_sum)

    return SimpleNamespace(
        abswidth=abswidth.item(),
        effwidth=effwidth.item(),
        area=area.item(),
        dtycyc=dtycyc.item(),
        max_pw=dtycyc.item(),
        num=1,
        max_b1=max_b1.item(),
        max_int_b1_sqr=max_int_b1_sqr.item(),
        max_rms_b1=max_rms_b1.item(),
        nom_fa=flip_angle.item(),  # flip angle [deg]
        nom_pw=duration.item() * 1e6,  # duration [us]
        nom_bw=nom_bw.item(),  # bandwidth [Hz]
        isodelay=isodelay * 1e6,  # isodelay in [us]
    )
