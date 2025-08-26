"""Demo sequence for testing and benchmark."""

__all__ = ["MPRAGE", "mprage"]

import math
import typing

import numpy as np
import pypulseq as pp

from pulserver_interpreter.pulseq import PulseqDesign

Sequence = typing.NewType("Sequence", None)


class MPRAGE(PulseqDesign):
    """
    Demo mprage sequence design class.

    Attributes
    ----------
    system : Opts, optional
        System limits or configuration. Default is ``None`` (Pulseq default).
    fov: tuple[float], optional
        Default field of view ``(dx, dy, dz)`` in ``[m]``.
        Default is ``(256e-3, 256e-3, 256e-3)``.
    mtx: tuple[int], optional
        Default matrix size ``(nx, ny, nz)``.
        Default is ``(256, 256, 256)``.
    TI: float, optional
        Default Inversion Time in ``[s]``.
        Default is ``140e-3``.
    TR: float, optional
        Default FLASH Repetition Time in ``[s]``.
        Default is ``10e-3``.
    T_recovery: float, optional
        Default Recovery Time in ``[s]``.
        Default is ``1e-3``.
    flip_angle: float, optional
        Default FLASH Flip Angle in ``[deg]``.
        Default is ``12.0``.
    dwell: float, optional
        Default ADC dwell time in ``[s]``.
        Default is ``1e-5``.

    Parameters
    ----------
    fov: tuple[float], optional
        Field of view ``(dx, dy, dz)`` in ``[m]``.
        Default is ``(256e-3, 256e-3, 256e-3)``.
    mtx: tuple[int], optional
        Matrix size ``(nx, ny, nz)``.
        Default is ``(256, 256, 256)``.
    TI: float, optional
        Inversion Time in ``[s]``.
        Default is ``140e-3``.
    TR: float, optional
        FLASH Repetition Time in ``[s]``.
        Default is ``10e-3``.
    T_recovery: float, optional
        Recovery Time in ``[s]``.
        Default is ``1e-3``.
    flip_angle: float, optional
        FLASH Flip Angle in ``[deg]``.
        Default is ``12.0``.
    dwell: float, optional
        ADC dwell time in ``[s]``.
        Default is ``1e-5``.

    Returns
    -------
    seq : Sequence
        Filled Sequence object.

    """

    def core(
        self,
        fov: tuple[float] = (256e-3, 256e-3, 256e-3),
        mtx: tuple[int] = (256, 256, 256),
        TI: float = 140e-3,
        TR: float = 10e-3,
        T_recovery: float = 1e-3,
        flip_angle: float = 12.0,
        dwell: float = 1e-5,
    ):
        return mpragecore(self, fov, mtx, TI, TR, T_recovery, flip_angle, dwell)


def mprage(
    system: pp.Opts = None,
    fov: tuple[float] = (256e-3, 256e-3, 256e-3),
    mtx: tuple[int] = (256, 256, 256),
    TI: float = 140e-3,
    TR: float = 10e-3,
    T_recovery: float = 1e-3,
    flip_angle: float = 12.0,
    dwell: float = 1e-5,
) -> Sequence:
    """
    Demo mprage sequence.

    Parameters
    ----------
    system : Opts, optional
        System limits or configuration. Default is ``None`` (Pulseq default).
    fov: tuple[float], optional
        Field of view ``(dx, dy, dz)`` in ``[m]``.
        Default is ``(256e-3, 256e-3, 256e-3)``.
    mtx: tuple[int], optional
        Matrix size ``(nx, ny, nz)``.
        Default is ``(256, 256, 256)``.
    TI: float, optional
        Inversion Time in ``[s]``.
        Default is ``140e-3``.
    TR: float, optional
        FLASH Repetition Time in ``[s]``.
        Default is ``10e-3``.
    T_recovery: float, optional
        Recovery Time in ``[s]``.
        Default is ``1e-3``.
    flip_angle: float, optional
        FLASH Flip Angle in ``[deg]``.
        Default is ``12.0``.
    dwell: float, optional
        ADC dwell time in ``[s]``.
        Default is ``1e-5``.

    Returns
    -------
    seq : Sequence
        Filled Sequence object.

    """
    mprage_design = MPRAGE(system)
    mprage_design.mode = "rt"
    return mprage_design(fov, mtx, TI, TR, T_recovery, flip_angle, dwell)


def mpragecore(
    self: PulseqDesign,
    fov: tuple[float] = (256e-3, 256e-3, 256e-3),
    mtx: tuple[int] = (256, 256, 256),
    TI: float = 140e-3,
    TR: float = 10e-3,
    T_recovery: float = 1e-3,
    flip_angle: float = 12.0,
    dwell: float = 1e-5,
) -> Sequence:
    """
    Actual mprage design routine.

    This is very close to a standard Pulseq sequence.

    Parameters
    ----------
    fov: tuple[float]
        Field of view ``(dx, dy, dz)`` in ``[m]``.
    mtx: tuple[int]
        Matrix size ``(nx, ny, nz)``.
    TI: float
        Inversion Time in ``[s]``.
    TR: float
        Repetition Time in ``[s]``.
    T_recovery: float
        Recovery Time in ``[s]``.
    flip_angle: float
        Flip Angle in ``[deg]``.
    dwell: float, optional
        ADC dwell time in ``[s]``.
        Default is ``1e-5``.

    """
    # Hardcoded parameters
    rf_spoiling_inc = 117
    gamma = 42.576e6

    # Initialize sequence
    seq = self.seq   # Standard Pulseq: seq = pp.Sequence()
    prot = self.mrd  # Sidecar MRD object

    # get system
    system = seq.system

    # Matrix
    Nx, Ny, Nz = mtx

    # Field of view
    fov_x, fov_y, fov_z = fov
    
    # ================ Set MRD protocol ==================
    prot.set_encoding(self.seqID)
    
    # save resonance fequency
    prot.set_h1_frequency(seq.system.B0)
    
    # spatial encoding
    prot.set_trajectory('cartesian')
    prot.set_fov(size=list(fov), osf=(1.0, 1.0, 1.0))
    prot.set_mtx(size=list(mtx), osf=(1.0, 1.0, 1.0))
    prot.set_kspace(axis='k0', min=0, max=Nx, center=None)
    prot.set_kspace(axis='k1', min=0, max=Ny, center=None)
    prot.set_kspace(axis='k2', min=0, max=Nz, center=None)
    prot.set_user_param('SliceThickness', float(fov_z / Nz))
    
    # contrast encoding
    prot.set_flip_angle_deg(flip_angle)
    # prot.set_echo_time(TE)
    prot.set_repetition_time(TR)
    prot.set_inversion_time(TI)
    prot.set_user_params('flipAn')
    
    

    # =========
    # RF preparatory, excitation
    # =========
    flip_exc = np.deg2rad(flip_angle)
    rf = pp.make_block_pulse(
        flip_angle=flip_exc, system=system, duration=250e-6, time_bw_product=4
    )

    flip_prep = math.pi / 2
    rf_prep = pp.make_block_pulse(
        flip_angle=flip_prep, system=system, duration=500e-6, time_bw_product=4
    )

    # =========
    # Readout
    # =========
    delta_kx = 1 / fov_x
    kx_width = Nx * delta_kx
    readout_time = Nx * dwell
    gx = pp.make_trapezoid(
        channel="x", system=system, flat_area=kx_width, flat_time=readout_time
    )
    adc = pp.make_adc(num_samples=Nx, duration=gx.flat_time, delay=gx.rise_time)

    # =========
    # Prephase and Rephase
    # =========
    delta_ky = 1 / fov_y
    delta_kz = 1 / fov_z
    phase_areas = (np.arange(Ny) - (Ny / 2)) * delta_ky
    slice_areas = (np.arange(Nz) - (Nz / 2)) * delta_kz

    gx_pre = pp.make_trapezoid(
        channel="x", system=system, area=-gx.area / 2, duration=2e-3
    )
    _gy_pre = pp.make_trapezoid(
        channel="y", system=system, area=phase_areas[-1], duration=2e-3
    )
    _gz_pre = pp.make_trapezoid(
        channel="z", system=system, area=slice_areas[-1], duration=2e-3
    )

    # =========
    # Spoilers
    # =========
    pre_time = 6.4e-4
    gx_spoil = pp.make_trapezoid(
        channel="x",
        system=system,
        area=(4 * np.pi) / (gamma * delta_kx * 1e-3) * gamma,
        duration=pre_time * 6,
    )
    gy_spoil = pp.make_trapezoid(
        channel="y",
        system=system,
        area=(4 * np.pi) / (gamma * delta_ky * 1e-3) * gamma,
        duration=pre_time * 6,
    )
    gz_spoil = pp.make_trapezoid(
        channel="z",
        system=system,
        area=(4 * np.pi) / (gamma * delta_kz * 1e-3) * gamma,
        duration=pre_time * 6,
    )

    # =========
    # Extended trapezoids: gx, gx_spoil
    # =========
    t_gx_extended = np.array(
        [
            0,
            gx.rise_time,
            gx.flat_time,
            (gx.rise_time * 2) + gx.flat_time + gx.fall_time,
        ]
    )
    amp_gx_extended = np.array([0, gx.amplitude, gx.amplitude, gx_spoil.amplitude])
    t_gx_spoil_extended = np.array(
        [
            0,
            gx_spoil.rise_time + gx_spoil.flat_time,
            gx_spoil.rise_time + gx_spoil.flat_time + gx_spoil.fall_time,
        ]
    )
    amp_gx_spoil_extended = np.array([gx_spoil.amplitude, gx_spoil.amplitude, 0])

    gx_extended = pp.make_extended_trapezoid(
        channel="x", times=t_gx_extended, amplitudes=amp_gx_extended
    )
    gx_spoil_extended = pp.make_extended_trapezoid(
        channel="x", times=t_gx_spoil_extended, amplitudes=amp_gx_spoil_extended
    )

    # =========
    # Delays
    # =========
    delay_TI = TI - pp.calc_duration(rf_prep) / 2 - pp.calc_duration(gx_spoil)
    delay_TR = (
        TR
        - pp.calc_duration(rf)
        - pp.calc_duration(gx_pre)
        - pp.calc_duration(gx)
        - pp.calc_duration(gx_spoil)
    )

    # Prepare delay events
    wait_TI = pp.make_delay(delay_TI)
    wait_TR = pp.make_delay(delay_TR)
    wait_recovery = pp.make_delay(T_recovery)

    # Pre-register objects that do not change while looping
    result = seq.register_grad_event(gx_spoil)
    gx_spoil.id = result if isinstance(result, int) else result[0]

    result = seq.register_grad_event(gy_spoil)
    gy_spoil.id = result if isinstance(result, int) else result[0]

    result = seq.register_grad_event(gz_spoil)
    gz_spoil.id = result if isinstance(result, int) else result[0]

    result = seq.register_grad_event(gx_pre)
    gx_pre.id = result if isinstance(result, int) else result[0]

    result = seq.register_grad_event(gx_extended)
    gx_extended.id = result if isinstance(result, int) else result[0]

    result = seq.register_grad_event(gx_spoil_extended)
    gx_spoil_extended.id = result if isinstance(result, int) else result[0]

    # Phase of the excitation RF object will change, therefore we only pre-register the shapes
    rf_prep.id, rf_prep.shape_IDs = seq.register_rf_event(rf_prep)
    _, rf.shape_IDs = seq.register_rf_event(rf)

    # Labels
    MAIN_SEQ = pp.make_label(
        type="SET", label="TRID", value=self.seqID
    )  # Standard Pulseq: pp.make_label(type="SET", label="TRID", value=1)
    TR_BREAK = pp.make_label(type="SET", label="TRID", value=-1)

    # Get scaling factors
    phase_scaling = phase_areas / (_gy_pre.amplitude + 1e-12)
    slice_scaling = slice_areas / (_gz_pre.amplitude + 1e-12)

    # Prepare
    for i in self.range(Ny):  # Standard Pulseq: for i in range(Ny)
        rf_phase = 0
        rf_inc = 0

        gy_pre = pp.scale_grad(_gy_pre, phase_scaling[i])
        gy_reph = pp.scale_grad(gy_pre, -1)

        # Pre-register PE events that repeat in the inner loop
        gy_pre.id = seq.register_grad_event(gy_pre)
        gy_reph.id = seq.register_grad_event(gy_reph)

        seq.add_block(rf_prep, MAIN_SEQ)
        seq.add_block(gx_spoil, gy_spoil, gz_spoil)
        seq.add_block(wait_TI)

        for j in range(Nz):
            rf.phase_offset = np.deg2rad(rf_phase)
            adc.phase_offset = np.deg2rad(rf_phase)

            gz_pre = pp.scale_grad(_gz_pre, slice_scaling[j])
            gz_reph = pp.scale_grad(gz_pre, -1)

            seq.add_block(rf, TR_BREAK)
            seq.add_block(gx_pre, gy_pre, gz_pre)
            seq.add_block(gx_extended, adc)
            seq.add_block(gx_spoil_extended, gy_reph, gz_reph)
            seq.add_block(wait_TR)
            
            # update header
            prot.append_acquisition(k1=i, k2=j)

            # update increment
            rf_inc = np.mod(rf_inc + rf_spoiling_inc, 360.0)
            rf_phase = np.mod(rf_phase + rf_inc, 360.0)

        seq.add_block(wait_recovery, TR_BREAK)

    return seq
