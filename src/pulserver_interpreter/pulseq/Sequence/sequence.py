"""Drop-in replacement for pypulseq.Sequence with TR/segment tracking"""

__all__ = ["Sequence"]

import datetime
from types import SimpleNamespace

import numpy as np

from pypulseq import Opts
from pypulseq import Sequence as PyPulseqSequence

from . import block
from . import segment


class Sequence:
    """Drop-in replacement for pypulseq.Sequence, supporting prep mode, TR/segment tracking."""

    def __init__(self, system: Opts | None = None, use_block_cache: bool = True):
        """
        Initialize a new Sequence instance.

        Parameters
        ----------
        system : Opts, optional
            System limits or configuration.
        use_block_cache : bool, optional
            Whether to use block cache (default: True).
        """
        self._system = system
        self.use_block_cache = use_block_cache
        self.clear()

    def clear(self):
        """
        Reset internal structure.
        """
        self._mode = "dry"  # 'dry', 'prep', 'eval', or 'rt'
        self.seq = PyPulseqSequence(
            system=self._system, use_block_cache=self.use_block_cache
        )

        # --- TR/block tracking ---
        self.first_tr_instances_trid_labels = (
            {}
        )  # dict: TRID -> first TR instance block labels
        self.current_trid = None
        self.current_block = 0
        self.within_tr = 0  # position within current TR

        # --- Global arrays ---
        self.block_tr_starts = []
        self.block_trid = []
        self.block_within_tr = []
        self.block_segment_id = None
        self.block_within_segment = None
        self.block_id = None

        # --- Segment/Block libraries ---
        self.unique_blocks = None
        self.segments = None
        self.trs = None

        # --- Status flags ---
        self.prepped = False
        self.evaluated = False

        # --- Initial parameters ---
        self.initial_tr_status = {}

        # --- Sequence info ----
        self.adc_count = 0
        self.n_total_segments = 0
        self.duration_cache = {}
        self._total_duration = 0.0

        # --- Real Time helpers ---
        self.start_block = 0
        self.end_block = np.inf

    def clear_buffer(self):
        self.seq = PyPulseqSequence(
            system=self._system, use_block_cache=self.use_block_cache
        )
        self.current_block = 0
        self.start_block = 0
        self.end_block = np.inf

    @property
    def system(self):
        return self.seq.system

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value: str):
        if value not in ["dry", "prep", "eval", "rt"]:
            raise ValueError(f"Mode (={value}) must be 'dry', 'prep', 'eval', or 'rt'.")
        self._mode = value

    @property
    def duration(self):
        return f"{datetime.timedelta(seconds=self._total_duration)}".split(".")[0]

    def add_block(self, *args) -> None:
        """Add a block to the sequence, dispatching to the appropriate method based on mode."""
        if self.mode == "dry" or self.mode == "prep":
            block.add_block_prep(self, *args)
        elif self.mode == "eval":
            block.add_block_eval(self, *args)
        elif self.mode == "rt":
            block.add_block_rt(self, *args)
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

    def check_timing(
        self, print_errors: bool = False  # noqa: ARG002
    ) -> tuple[bool, list[SimpleNamespace]]:
        return True, []

    def register_adc_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_adc_event(event)

    def register_control_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_control_event(event)

    def register_grad_event(self, event: SimpleNamespace) -> int | tuple[int, int]:
        return self.seq.register_grad_event(event)

    def register_label_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_label_event(event)

    def register_rf_event(self, event: SimpleNamespace) -> tuple[int, list[int]]:
        return self.seq.register_rf_event(event)

    def register_rf_shim_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_rf_shim_event(event)

    def register_rotation_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_rotation_event(event)

    def register_soft_delay_event(self, event: SimpleNamespace) -> int:
        return self.seq.register_soft_delay_event(event)

    def paper_plot(
        self,
        time_range: tuple[float] = (0, np.inf),
        line_width: float = 1.2,
        axes_color: tuple[float] = (0.5, 0.5, 0.5),
        rf_color: str = "black",
        gx_color: str = "blue",
        gy_color: str = "red",
        gz_color: tuple[float] = (0, 0.5, 0.3),
        rf_plot: str = "abs",
    ):
        pass  # dummy method

    def plot(
        self,
        label: str = str(),
        show_blocks: bool = False,
        save: bool = False,
        time_range=(0, np.inf),
        time_disp: str = "s",
        grad_disp: str = "kHz/m",
        plot_now: bool = True,
    ) -> None:
        pass  # dummy method

    def set_definition(
        self, key: str, value: float | int | list | np.ndarray | str | tuple
    ) -> None:
        pass  # dummy method

    def test_report(self) -> str:
        return ""  # dummy method

    def write(
        self,
        name: str,
        create_signature: bool = True,
        remove_duplicates: bool = True,
        check_timing: bool = True,
    ) -> str | None:
        pass  # dummy method

    def _get_tr(self, idx):
        tr = self.trs[idx].blocks
        seq = PyPulseqSequence(self.system)
        init_status = self.initial_tr_status[idx]

        # Fill sequence
        for block_id in tr:
            block = self.unique_blocks.get_block(block_id)
            seq.add_block(block)

        # Set duration, amplitudes and remove extension
        for n in range(tr.size):
            seq.block_events[n + 1][0] = init_status[n, 0]

            rf_id = seq.block_events[n + 1][1]
            if rf_id:
                tmp = np.asarray(seq.rf_library.data[rf_id])
                tmp[0] = init_status[n, 1]
                seq.rf_library.data[rf_id] = tuple(tmp)

            gx_id = seq.block_events[n + 1][2]
            if gx_id:
                tmp = np.asarray(seq.grad_library.data[gx_id])
                tmp[0] = init_status[n, 2]
                seq.grad_library.data[gx_id] = tuple(tmp)

            gy_id = seq.block_events[n + 1][3]
            if gy_id:
                tmp = np.asarray(seq.grad_library.data[gy_id])
                tmp[0] = init_status[n, 3]
                seq.grad_library.data[gy_id] = tuple(tmp)

            gz_id = seq.block_events[n + 1][4]
            if gz_id:
                tmp = np.asarray(seq.grad_library.data[gz_id])
                tmp[0] = init_status[n, 4]
                seq.grad_library.data[gz_id] = tuple(tmp)

            seq.block_events[n + 1][6] = 0

        # Set phase/freq offsets for rf, adc to 0
        for n in seq.rf_library.data:
            tmp = np.asarray(seq.rf_library.data[n])
            tmp[6:10] = 0  # (freq_ppm, phase_ppm, freq_off, phase_off)
            seq.rf_library.data[n] = tuple(tmp)
        for n in seq.adc_library.data:
            tmp = np.asarray(seq.adc_library.data[n])
            tmp[3:8] = 0  # (freq_ppm, phase_ppm, freq_off, phase_off, phase_mod_id)
            seq.adc_library.data[n] = tuple(tmp)

        return seq

    def tr(self, idx: int | None = None) -> PyPulseqSequence | dict:
        """
        Get desired TR as a PyPulseq sequence.

        Parameters
        ----------
        idx : int, optional
            TR index. If not provided, returns the dicionary
            containing all TRs. The default is None.

        """
        if idx is not None:
            return self._get_tr(idx)
        return {idx: self._get_tr(idx) for idx in self.trs}

    def _get_segment(self, idx):
        segment = self.segments[idx]
        seq = PyPulseqSequence(self.system)
        init_status = self.initial_segment_status[idx]

        # Fill sequence
        for block_id in segment:
            block = self.unique_blocks.get_block(block_id)
            seq.add_block(block)

        # Set duration, amplitudes and remove extension
        for n in range(segment.size):
            seq.block_events[n + 1][0] = init_status[n, 0]

            rf_id = seq.block_events[n + 1][1]
            if rf_id:
                tmp = np.asarray(seq.rf_library.data[rf_id])
                tmp[0] = init_status[n, 1]
                seq.rf_library.data[rf_id] = tuple(tmp)

            gx_id = seq.block_events[n + 1][2]
            if gx_id:
                tmp = np.asarray(seq.grad_library.data[gx_id])
                tmp[0] = init_status[n, 2]
                seq.grad_library.data[gx_id] = tuple(tmp)

            gy_id = seq.block_events[n + 1][3]
            if gy_id:
                tmp = np.asarray(seq.grad_library.data[gy_id])
                tmp[0] = init_status[n, 3]
                seq.grad_library.data[gy_id] = tuple(tmp)

            gz_id = seq.block_events[n + 1][4]
            if gz_id:
                tmp = np.asarray(seq.grad_library.data[gz_id])
                tmp[0] = init_status[n, 4]
                seq.grad_library.data[gz_id] = tuple(tmp)

            seq.block_events[n + 1][6] = 0

        # Set phase/freq offsets for rf, adc to 0
        for n in seq.rf_library.data:
            tmp = np.asarray(seq.rf_library.data[n])
            tmp[6:10] = 0  # (freq_ppm, phase_ppm, freq_off, phase_off)
            seq.rf_library.data[n] = tuple(tmp)
        for n in seq.adc_library.data:
            tmp = np.asarray(seq.adc_library.data[n])
            tmp[3:8] = 0  # (freq_ppm, phase_ppm, freq_off, phase_off, phase_mod_id)
            seq.adc_library.data[n] = tuple(tmp)

        return seq

    def segment(self, idx: int | None = None) -> PyPulseqSequence | dict:
        """
        Get desired Segment as a PyPulseq sequence.

        Parameters
        ----------
        idx : int, optional
            Segment index. If not provided, returns the dicionary
            containing all segments. The default is None.

        """
        if idx is not None:
            return self._get_segment(idx)
        return {idx: self._get_segment(idx) for idx in self.segments}

    def get_seq_structure(self):
        """
        Use the external segment.build_segments wrapper to perform block deduplication, segment splitting, and segment deduplication.
        Stores the resulting segment library, mapping arrays, and block library as attributes.
        """
        if self.prepped:
            raise ValueError(
                "Sequence is already prepared. Please call clear() before parsing structure again."
            )
        (
            self.trs,
            self.segments,
            self.unique_blocks,
            self.block_trid,
            self.block_within_tr,
            self.block_segment_id,
            self.block_within_segment,
            self.block_id,
            self.n_total_segments,
        ) = segment.get_seq_structure(
            self.seq, self.first_tr_instances_trid_labels, self.block_trid
        )

        # Preallocate initial TR status
        self.initial_tr_status = {
            k: np.zeros((v.blocks.size, 5), dtype=float) for k, v in self.trs.items()
        }  # (dur, rf, gx, gy, gz)
        for k in self.initial_tr_status:
            self.initial_tr_status[k][:, 0] = np.inf
        self.tr_gradient_signs = {
            k: np.zeros((v.blocks.size, 3), dtype=int) for k, v in self.trs.items()
        }  # (x, y, z)

        # Preallocate initial segment status
        self.initial_segment_status = {
            k: np.zeros((v.size, 5), dtype=float) for k, v in self.segments.items()
        }  # (dur, rf, gx, gy, gz)
        for k in self.initial_segment_status:
            self.initial_segment_status[k][:, 0] = np.inf
        self.segment_gradient_signs = {
            k: np.zeros((v.size, 3), dtype=int) for k, v in self.segments.items()
        }  # (x, y, z)

        self.prepped = True

    def get_initial_status(self):
        if not self.prepped:
            raise RuntimeError("Eval mode requires a valid sequence structure.")

        if self.evaluated:
            raise ValueError(
                "Sequence is already evaluated. Please call clear() before parsing initial TR status again."
            )
        for k in self.initial_tr_status:
            self.initial_tr_status[k][:, 2] *= self.tr_gradient_signs[k][:, 0]
            self.initial_tr_status[k][:, 3] *= self.tr_gradient_signs[k][:, 1]
            self.initial_tr_status[k][:, 4] *= self.tr_gradient_signs[k][:, 2]
        self.tr_gradient_signs = None
        for k in self.initial_segment_status:
            self.initial_segment_status[k][:, 2] *= self.segment_gradient_signs[k][:, 0]
            self.initial_segment_status[k][:, 3] *= self.segment_gradient_signs[k][:, 1]
            self.initial_segment_status[k][:, 4] *= self.segment_gradient_signs[k][:, 2]
        self.segment_gradient_signs = None
        self.evaluated = True
