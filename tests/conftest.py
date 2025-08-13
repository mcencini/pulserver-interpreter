"""Auxiliary test case function."""

import math

import numpy as np
import pypulseq as pp
import pytest

nz = 256

@pytest.fixture
def Nz():
    return nz

@pytest.fixture
def mprage():
    def fill_mprage_seq(seq):
        """
        Fill the given pulserver Sequence object with an MPRAGE-like loop using base_blocks.
        This function is agnostic to the mode of the Sequence (prep/eval).
        """
        # get system
        system = seq.system

        # Size
        Nx = 256
        Ny = 256
        Nz = nz

        # Parameters
        fov = 256e-3
        fov_z = 256e-3
        slice_thickness = 1e-3
        section_thickness = 5e-3

        # =========
        # RF preparatory, excitation
        # =========
        flip_exc = 12 * math.pi / 180
        rf = pp.make_block_pulse(flip_angle=flip_exc, system=system, duration=250e-6, time_bw_product=4)

        flip_prep = 90 * math.pi / 180
        rf_prep = pp.make_block_pulse(flip_angle=flip_prep, system=system, duration=500e-6, time_bw_product=4)

        # =========
        # Readout
        # =========
        delta_k = 1 / fov
        k_width = Nx * delta_k
        readout_time = 3.5e-3
        gx = pp.make_trapezoid(channel='x', system=system, flat_area=k_width, flat_time=readout_time)
        adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)

        # =========
        # Prephase and Rephase
        # =========
        delta_kz = 1 / fov_z
        phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_k
        slice_areas = (np.arange(Nz) - (Nz / 2)) * delta_kz

        gx_pre = pp.make_trapezoid(channel='x', system=system, area=-gx.area / 2, duration=2e-3)
        gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[-1], duration=2e-3)

        # =========
        # Spoilers
        # =========
        pre_time = 6.4e-4
        gx_spoil = pp.make_trapezoid(
            channel='x',
            system=system,
            area=(4 * np.pi) / (42.576e6 * delta_k * 1e-3) * 42.576e6,
            duration=pre_time * 6,
        )
        gy_spoil = pp.make_trapezoid(
            channel='y',
            system=system,
            area=(4 * np.pi) / (42.576e6 * delta_k * 1e-3) * 42.576e6,
            duration=pre_time * 6,
        )
        gz_spoil = pp.make_trapezoid(
            channel='z',
            system=system,
            area=(4 * np.pi) / (42.576e6 * delta_kz * 1e-3) * 42.576e6,
            duration=pre_time * 6,
        )

        # =========
        # Extended trapezoids: gx, gx_spoil
        # =========
        t_gx_extended = np.array([0, gx.rise_time, gx.flat_time, (gx.rise_time * 2) + gx.flat_time + gx.fall_time])
        amp_gx_extended = np.array([0, gx.amplitude, gx.amplitude, gx_spoil.amplitude])
        t_gx_spoil_extended = np.array(
            [
                0,
                gx_spoil.rise_time + gx_spoil.flat_time,
                gx_spoil.rise_time + gx_spoil.flat_time + gx_spoil.fall_time,
            ]
        )
        amp_gx_spoil_extended = np.array([gx_spoil.amplitude, gx_spoil.amplitude, 0])

        gx_extended = pp.make_extended_trapezoid(channel='x', times=t_gx_extended, amplitudes=amp_gx_extended)
        gx_spoil_extended = pp.make_extended_trapezoid(channel='x', times=t_gx_spoil_extended, amplitudes=amp_gx_spoil_extended)

        # =========
        # Delays
        # =========
        TE, TI, TR, T_recovery = 4e-3, 140e-3, 10e-3, 1e-3
        delay_TE = TE - pp.calc_duration(rf) / 2 - pp.calc_duration(gx_pre) - pp.calc_duration(gx) / 2
        delay_TI = TI - pp.calc_duration(rf_prep) / 2 - pp.calc_duration(gx_spoil)
        delay_TR = TR - pp.calc_duration(rf) - pp.calc_duration(gx_pre) - pp.calc_duration(gx) - pp.calc_duration(gx_spoil)

        for i in range(Ny):
            gy_pre = pp.make_trapezoid(channel='y', system=system, area=phase_areas[i], duration=2e-3)

            seq.add_block(rf_prep, pp.make_label(type='SET', label='TRID', value=1))
            seq.add_block(gx_spoil, gy_spoil, gz_spoil)
            seq.add_block(pp.make_delay(delay_TI))

            for j in range(Nz):
                gz_pre = pp.make_trapezoid(channel='z', system=system, area=slice_areas[j], duration=2e-3)
                gz_reph = pp.make_trapezoid(channel='z', system=system, area=-slice_areas[j], duration=2e-3)

                seq.add_block(rf, pp.make_label(type='SET', label='TRID', value=-1))
                seq.add_block(gx_pre, gy_pre, gz_pre)
                seq.add_block(gx_extended, adc)
                seq.add_block(gx_spoil_extended, gz_reph)
                seq.add_block(pp.make_delay(delay_TR))

            seq.add_block(pp.make_delay(T_recovery), pp.make_label(type='SET', label='TRID', value=-1))

        return seq

    return fill_mprage_seq
