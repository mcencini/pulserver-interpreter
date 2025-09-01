"""Test sequence builder."""

from pulserver_interpreter.builder import (
    BaseBuilder,
    CompositeBuilder,
    InterleavedBuilder,
    SequenceRole,
)


# ------------------------
# Dummy builders for tests
# ------------------------
class DummySeq(BaseBuilder):
    """Leaf builder that emits a simple list of integers for each iteration."""

    def prepare(self, events=None, niter=1):
        self.events = events if events is not None else [1]
        self.Niter = niter

    def encoding_space(self):
        return {
            "role": self.role.value,
            "fov": [1, 1, 1],
            "matrix": (1, 1, 1),
            "limits": {},
            "flags": self.role.flag,
        }

    def kernel(self, seq=None, prot=None, idx=0):
        """Append one event per iteration to seq or prot."""
        val = self.events[idx % len(self.events)]
        if seq is not None:
            if not hasattr(seq, "data"):
                seq.data = []
            seq.data.append(val)
        if prot is not None:
            prot.append(val)


# ------------------------
# Helper mock sequence
# ------------------------
class MockSeq:
    """Simple container for Pulseq-like events."""

    def __init__(self):
        self.data = []


# ------------------------
# Standalone builder tests
# ------------------------
def test_dummyseq_emission_seq_and_prot():
    seq_obj = DummySeq(events=[10, 20], niter=3)
    seq = MockSeq()
    prot = []
    seq, prot = seq_obj(seq=seq, prot=prot)
    # seq and prot should match
    assert seq.data == [10, 20, 10]
    assert prot == [10, 20, 10]


# ------------------------
# Composite builder tests
# ------------------------
class CompositeTest(CompositeBuilder):
    first: DummySeq
    second: DummySeq

    def prepare(self):
        self.first = DummySeq(role=SequenceRole.CALIBRATION, events=[1, 2], niter=2)
        self.second = DummySeq(role=SequenceRole.MAIN, events=[3, 4], niter=2)

    def encoding_space(self):
        return {}


def test_composite_seq_and_prot():
    comp = CompositeTest()
    seq = MockSeq()
    prot = []
    seq, prot = comp(seq=seq, prot=prot)
    # concatenated order
    assert seq.data == [1, 2, 3, 4]
    assert prot == [1, 2, 3, 4]
    # children and encoding indices
    children = comp.children()
    assert children == [comp.first, comp.second]
    assert comp.first.enc_idx == 0
    assert comp.first.role == SequenceRole.CALIBRATION
    assert comp.second.enc_idx == 1
    assert comp.second.role == SequenceRole.MAIN


# ------------------------
# Interleaved builder tests
# ------------------------
class InterleavedTest(InterleavedBuilder):
    child1: DummySeq
    child2: DummySeq

    def prepare(self):
        self.child1 = DummySeq(role=SequenceRole.MAIN, events=[1, 2, 3])
        self.child2 = DummySeq(role=SequenceRole.NAVIGATION, events=[4, 5, 6])
        self.Niter = 3

    def encoding_space(self):
        return {}

    def kernel(self, seq=None, prot=None, idx=0):
        self.child1.kernel(seq=seq, prot=prot, idx=idx)
        self.child2.kernel(seq=seq, prot=prot, idx=idx)


def test_interleaved_seq_and_prot():
    inter = InterleavedTest()
    seq = MockSeq()
    prot = []
    seq, prot = inter(seq=seq, prot=prot)
    # child1: [1,2,3], child2: [4,5,6], interleaved: [1,4,2,5,3,6]
    assert seq.data == [1, 4, 2, 5, 3, 6]
    assert prot == [1, 4, 2, 5, 3, 6]
    # check encoding indices
    assert inter.child1.enc_idx == 0
    assert inter.child1.role == SequenceRole.MAIN
    assert inter.child2.enc_idx == 1
    assert inter.child2.role == SequenceRole.NAVIGATION


# ------------------------
# Nested composite + interleaved
# ------------------------
class NestedInterleaved(InterleavedBuilder):
    main: DummySeq
    nav: DummySeq

    def prepare(self):
        self.main = DummySeq(role=SequenceRole.MAIN, events=[10, 20, 30])
        self.nav = DummySeq(role=SequenceRole.NAVIGATION, events=[100, 200, 300])
        self.Niter = 3

    def encoding_space(self):
        return {}

    def kernel(self, seq=None, prot=None, idx=0):
        self.main.kernel(seq=seq, prot=prot, idx=idx)
        self.nav.kernel(seq=seq, prot=prot, idx=idx)


class NestedComposite(CompositeBuilder):
    calib: DummySeq
    main_seq: NestedInterleaved

    def prepare(self):
        self.calib = DummySeq(role=SequenceRole.CALIBRATION, events=[1, 2], niter=2)
        self.main_seq = NestedInterleaved()

    def encoding_space(self):
        return {}


def test_nested_composite_interleaved_seq_and_prot():
    nested = NestedComposite()
    seq = MockSeq()
    prot = []
    seq, prot = nested.emit(seq=seq, prot=prot)
    # expected emission: calib [1,2] + interleaved [10,100,20,200,30,300]
    assert seq.data == [1, 2, 10, 100, 20, 200, 30, 300]
    assert prot == [1, 2, 10, 100, 20, 200, 30, 300]
    # encoding indices
    assert nested.calib.enc_idx == 0
    assert nested.calib.role == SequenceRole.CALIBRATION
    assert nested.main_seq.main.enc_idx == 1
    assert nested.main_seq.main.role == SequenceRole.MAIN
    assert nested.main_seq.nav.enc_idx == 2
    assert nested.main_seq.nav.role == SequenceRole.NAVIGATION
