"""Add block replacements."""

__all__ = ["add_block_eval", "add_block_prep", "add_block_rt"]


import numpy as np

from pypulseq import calc_duration


def add_block_prep(self, *args) -> None:
    """Add a block in preparation mode, tracking TRID, within-TR index, and first TR instance labels."""
    if self.prepped:
        raise ValueError(
            "Sequence is already prepared. Please call clear() running another preparation pass."
        )

    trid_label = 0
    for obj in args:
        if hasattr(obj, "label") and getattr(obj, "label", None) == "TRID":
            trid_label = getattr(obj, "value", 0)
            break

    # If TRID > 0 and not already in definitions, start new TRID definition
    if trid_label > 0:
        self.within_tr = 0
        if trid_label not in self.first_tr_instances_trid_labels:
            self.first_tr_instances_trid_labels[trid_label] = []
            self.current_trid = trid_label
        else:
            self.current_trid = None

    # Only update definition and add blocks if we are building the first instance
    if self.current_trid is not None:
        self.first_tr_instances_trid_labels[self.current_trid].append(trid_label)
        self.seq.add_block(*args)

    # Update global arrays for every block
    self._total_duration += calc_duration(*args)
    self.n_total_segments += trid_label != 0
    self.block_trid.append(trid_label)
    self.block_within_tr.append(self.within_tr)
    self.within_tr += 1


def add_block_eval(self, *args) -> None:
    """Add a block in evaluation mode, keeping max rf and gradient amplitudes and minimum duration."""
    if not self.prepped:
        raise RuntimeError("Eval mode requires a valid sequence structure.")

    if self.evaluated:
        raise ValueError(
            "Sequence is already evaluated. Please call clear() running another evaluation pass."
        )

    # Default values
    duration = 0.0
    rf_amp = 0.0
    gx_amp = 0.0
    gy_amp = 0.0
    gz_amp = 0.0
    gx_sign = 1
    gy_sign = 1
    gz_sign = 1

    # parse duration, rf_amp, grad amplitudes and sign
    for obj in args:
        if isinstance(obj, float):
            duration = max(duration, obj)
            continue
        typ = getattr(obj, "type", None)
        if typ == "rf":
            rf_amp = np.max(np.abs(obj.signal))
            duration = max(duration, obj.delay + obj.shape_dur)
        elif typ == "trap":
            if obj.channel == "x":
                gx_amp = np.abs(obj.amplitude)
                gx_sign = np.sign(obj.amplitude)
            elif obj.channel == "y":
                gy_amp = np.abs(obj.amplitude)
                gy_sign = np.sign(obj.amplitude)
            elif obj.channel == "z":
                gz_amp = np.abs(obj.amplitude)
                gz_sign = np.sign(obj.amplitude)
            duration = max(
                duration,
                obj.delay
                + obj.rise_time
                + obj.flat_time
                + obj.fall_time
                + obj.fall_time,
            )
        elif typ == "grad":
            maxval = np.argmax(np.abs(obj.waveform))
            if obj.channel == "x":
                gx_amp = np.abs(obj.waveform[maxval])
                gx_sign = np.sign(obj.waveform[maxval])
            elif obj.channel == "y":
                gy_amp = np.abs(obj.waveform[maxval])
                gy_sign = np.sign(obj.waveform[maxval])
            elif obj.channel == "z":
                gz_amp = np.abs(obj.waveform[maxval])
                gz_sign = np.sign(obj.waveform[maxval])
            duration = max(duration, obj.delay + obj.shape_dur)
        elif typ == "adc":
            self.adc_count += 1
            if hasattr(obj, "duration"):
                duration = max(duration, obj.delay + obj.duration)
            else:
                duration = max(duration, obj.delay + obj.num_samples * obj.dwell)
        elif typ == "delay":
            duration = max(duration, obj.delay)

    # Get current TR ID
    trid = self.block_trid[self.current_block]
    idx = self.block_within_tr[self.current_block]

    # Assign amplitudes
    self.initial_tr_status[trid][idx, 0] = min(
        self.initial_tr_status[trid][idx, 0], duration
    )
    self.initial_tr_status[trid][idx, 1] = max(
        self.initial_tr_status[trid][idx, 1], rf_amp
    )
    self.initial_tr_status[trid][idx, 2] = max(
        self.initial_tr_status[trid][idx, 2], gx_amp
    )
    self.initial_tr_status[trid][idx, 3] = max(
        self.initial_tr_status[trid][idx, 3], gy_amp
    )
    self.initial_tr_status[trid][idx, 4] = max(
        self.initial_tr_status[trid][idx, 4], gz_amp
    )

    # Assign signs
    if self.tr_gradient_signs[trid][idx, 0] == 0:
        self.tr_gradient_signs[trid][idx, 0] = gx_sign
    if self.tr_gradient_signs[trid][idx, 1] == 0:
        self.tr_gradient_signs[trid][idx, 1] = gy_sign
    if self.tr_gradient_signs[trid][idx, 2] == 0:
        self.tr_gradient_signs[trid][idx, 2] = gz_sign

    # Get current Segment ID
    segment_id = self.block_segment_id[self.current_block]
    idx = self.block_within_segment[self.current_block]

    # Assign amplitudes
    self.initial_segment_status[segment_id][idx, 0] = min(
        self.initial_segment_status[segment_id][idx, 0], duration
    )
    self.initial_segment_status[segment_id][idx, 1] = max(
        self.initial_segment_status[segment_id][idx, 1], rf_amp
    )
    self.initial_segment_status[segment_id][idx, 2] = max(
        self.initial_segment_status[segment_id][idx, 2], gx_amp
    )
    self.initial_segment_status[segment_id][idx, 3] = max(
        self.initial_segment_status[segment_id][idx, 3], gy_amp
    )
    self.initial_segment_status[segment_id][idx, 4] = max(
        self.initial_segment_status[segment_id][idx, 4], gz_amp
    )

    # Assign signs
    if self.segment_gradient_signs[segment_id][idx, 0] == 0:
        self.segment_gradient_signs[segment_id][idx, 0] = gx_sign
    if self.segment_gradient_signs[segment_id][idx, 1] == 0:
        self.segment_gradient_signs[segment_id][idx, 1] = gy_sign
    if self.segment_gradient_signs[segment_id][idx, 2] == 0:
        self.segment_gradient_signs[segment_id][idx, 2] = gz_sign

    # Update position
    self.current_block += 1


def add_block_rt(self, *args):
    """Add a block in real-time mode."""
    if self.current_block >= self.start_block and self.current_block < self.end_block:
        self.seq.add_block(*args)
    self.current_block += 1
