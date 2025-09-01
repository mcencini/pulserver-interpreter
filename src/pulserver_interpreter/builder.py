"""
Blueprint for modular Pulseq/ISMRMRD sequence builders.

Design notes
----------
- Subclasses implement:
    - prepare(*args, **kwargs): build events, create child builders (if any), set self.Niter when required
    - kernel(seq, idx, prot=None): emit one iteration (use self.enc_idx for encoding index)
    - encoding_space(): return dict describing FOV/matrix/limits/role/flags
- BaseBuilder.__init__ automatically runs prepare(), then post_prepare(), then validate().
- MultiChildBuilder.post_prepare() runs recursive, left-to-right, global encoding-index assignment.
- CompositeBuilder skips checking for its own Niter (it simply emits children sequentially).
- InterleavedBuilder inherits MultiChildBuilder and uses the base validation (so prepare() must set Niter).

"""

from __future__ import annotations

__all__ = ["BaseBuilder", "CompositeBuilder", "InterleavedBuilder"]

from abc import ABC, abstractmethod
from enum import Enum

import pypulseq as pp
import ismrmrd as mrd


class SequenceRole(Enum):
    """
    Sequence roles used to derive ISMRMRD acquisition flags.

    Attributes
    ----------
    MAIN, CALIBRATION, NAVIGATION : SequenceRole
        Roles that builders can be assigned.
    """

    MAIN = "main"
    CALIBRATION = "calibration"
    NAVIGATION = "navigation"
    NOISE = "noise"
    COIL_CORR = "coilcorr"

    @property
    def flag(self) -> int:
        """
        Map role to an ISMRMRD acquisition flag.

        Returns
        -------
        int
            ISMRMRD acquisition flag for the role (0 if no special flag).
        """
        role_flag_map = {
            SequenceRole.MAIN: 0,
            SequenceRole.CALIBRATION: mrd.ACQ_IS_PARALLEL_CALIBRATION,
            SequenceRole.NAVIGATION: mrd.ACQ_IS_NAVIGATION_DATA,
            SequenceRole.NOISE: mrd.ACQ_IS_NOISE_MEASUREMENT,
            SequenceRole.COIL_CORR: mrd.ACQ_IS_SURFACECOILCORRECTIONSCAN_DATA,
        }
        return role_flag_map[self]


class BaseBuilder(ABC):
    """
    Abstract base for every sequence builder.

    Parameters
    ----------
    *args, **kwargs :
        Arguments forwarded to `prepare()`.

    Attributes
    ----------
    role : SequenceRole
        Role assigned to this builder (default: MAIN)
    Niter : int | None
        Number of iterations for the builder's outer loop. Subclasses must
        set this in `prepare()` unless the subclass is a composite that
        intentionally does not use its own `Niter`.
    enc_idx : int | None
        Global encoding index assigned by parent composite; set to None by default.
    """

    role: SequenceRole = SequenceRole.MAIN
    enc_idx: int | None = None
    Niter: int | None = None

    def __init__(self, role: SequenceRole = SequenceRole.MAIN, *args, **kwargs):
        # store role, call prepare, allow subclass to consume args/kwargs
        self.role = role
        # subclass prepare builds events, sets Niter if needed, instantiates children
        self.prepare(*args, **kwargs)
        self.encoding_space()
        # hook for post-prepare work (e.g., assign enc_idx in multi-child builders)
        self.post_prepare()
        # final validation (by default enforce Niter is set)
        self.validate()

    def __call__(
        self, seq: pp.Sequence | None = None, prot: list[mrd.Acquisition] | None = None
    ) -> None:
        """Convenience: call the builder to emit into seq/prot."""
        return self.emit(seq=seq, prot=prot)

    @abstractmethod
    def prepare(self, *args, **kwargs) -> None:
        """
        Create RF/grad/adc events, instantiate child builders (if any), set `Niter`.

        Subclasses implement this with whatever signature they prefer (accepting *args/**kwargs).
        Must set `self.Niter` for leaf builders and interleaved builders.
        For composites that do not need an `Niter`, leave `Niter` as None.
        """
        ...

    @abstractmethod
    def encoding_space(self) -> dict:
        """
        Return encoding-space metadata for this builder.

        Returns
        -------
        dict
            Minimal expected keys: 'role' (string), 'fov' (list or tuple), 'matrix' (tuple),
            'limits' (dict). Include 'flags' if desired (or set from `self.role.flag`).
        """
        ...

    def post_prepare(self) -> None:
        """
        Optional hook executed after `prepare()`.

        MultiChildBuilder overrides this to perform global encoding-index assignment.
        Default implementation is a no-op.
        """
        return None

    def validate(self) -> None:
        """
        Default validation after prepare/post_prepare: ensure Niter exists.

        Leaf/interleaved builders normally must set Niter in prepare().
        Composite builders override `validate()` to skip this check.
        """
        if self.Niter is None:
            raise ValueError(
                f"{self.__class__.__name__}: Niter must be set in prepare()."
            )

    @abstractmethod
    def kernel(
        self,
        seq: pp.Sequence | None = None,
        prot: list[mrd.Acquisition] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Emit one repetition / TR of the sequence.

        Parameters
        ----------
        seq : pypulseq.Sequence or None
            Pulseq sequence to append blocks to (may be None in pure-metadata runs).
        idx : int
            Outer-loop index (0..Niter-1).
        prot : list[mrd.Acquisition] or None
            Optional list/collection to which ISMRMRD acquisition records should be appended.
        """
        ...

    def emit(
        self, seq: pp.Sequence | None = None, prot: list[mrd.Acquisition] | None = None
    ) -> None:
        """
        Default emission: loop over Niter and call kernel.

        Parameters
        ----------
        seq : pypulseq.Sequence | None
        prot : list | None
        """
        if self.Niter is None:
            # This helps catch leaf/interleaved builders that forgot to set Niter
            raise ValueError(
                f"{self.__class__.__name__}: Niter is not set. Must be set in prepare()."
            )
        for n in range(self.Niter):
            self.kernel(seq=seq, prot=prot, idx=n)

        return seq, prot


class MultiChildBuilder(BaseBuilder):
    """Base for builders that contain other builders (composite or interleaved).

    Child discovery is performed by reading the class' __annotations__ in
    left-to-right order and returning attributes that are instances of BaseBuilder.
    Subclasses must assign child attributes in `prepare()` before `post_prepare`
    returns so assignment and validation can run.
    """

    def children(self) -> list[BaseBuilder]:
        """
        Return child builder instances in declaration order.

        Notes
        -----
        This inspects the class __annotations__ keys in definition order and
        returns the attribute values that are BaseBuilder instances.
        """
        out: list[BaseBuilder] = []
        # preserve declaration order from __annotations__
        for name in getattr(self.__class__, "__annotations__", {}):
            val = getattr(self, name, None)
            if isinstance(val, BaseBuilder):
                out.append(val)
        return out

    def post_prepare(self) -> None:
        """
        After prepare(), assign global encoding indices recursively.

        The assignment is left-to-right over children and nested structures,
        guaranteeing globally unique `enc_idx` for leaf builders.
        """
        # start assigning from 0
        self._assign_encoding_indices(0)
        # allow subclasses to extend post_prepare if needed
        return None

    def _assign_encoding_indices(self, start_idx: int = 0) -> int:
        """
        Recursively walk children in order and assign unique enc_idx.

        Parameters
        ----------
        start_idx : int
            global index to start assigning from.

        Returns
        -------
        int
            next available global encoding index after assignment
        """
        for child in self.children():
            if isinstance(child, MultiChildBuilder):
                # nested composite/interleaved: let it assign its internal children
                start_idx = child._assign_encoding_indices(start_idx)
            else:
                # leaf builder: assign and increment
                child.enc_idx = start_idx
                start_idx += 1
        return start_idx


class CompositeBuilder(MultiChildBuilder):
    """Composite builder: executes children sequentially.

    Subclasses should implement `prepare()` and create/assign child attributes.
    CompositeBuilder intentionally skips `Niter` validation because it doesn't
    use a single outer-loop iteration count; each child manages its own `Niter`.
    """

    def validate(self) -> None:
        """Skip Niter check for composites (children manage their own iterations)."""
        return None

    def kernel(self) -> None:
        """No need to define custom kernel (sequentially run children kernels)."""
        return None

    def emit(
        self, seq: pp.Sequence | None = None, prot: list[mrd.Acquisition] | None = None
    ) -> None:
        """Emit all children sequentially (delegates to child.emit())."""
        for child in self.children():
            child(seq=seq, prot=prot)
        return seq, prot


class InterleavedBuilder(MultiChildBuilder):
    """
    Interleaved builder: children are interleaved during a single outer loop.

    Subclasses must set `self.Niter` in prepare() (and typically compute it from children),
    and implement `kernel()` to define what happens inside each outer iteration.
    The base `emit()` from BaseBuilder is used (loop over Niter calling kernel()).
    """
