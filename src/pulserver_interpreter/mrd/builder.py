"""MRD builder."""

__all__ = ["ISMRMRDBuilder"]

import copy
import warnings

from enum import Enum
from types import SimpleNamespace

import numpy as np

import ismrmrd as mrd
import ismrmrd.xsd as xsd

import pypulseq as pp


def __dummy_system__():
    sys = pp.Opts()
    sys.max_grad = np.inf
    sys.max_slew = np.inf
    return sys


DUMMY_SYSTEM = __dummy_system__()
OTHER = xsd.trajectoryType.OTHER


class ISMRMRDBuilder:
    def __init__(self, mode="static"):
        self.mode = mode  # 'dry', 'prep', 'eval', 'rt', 'static
        self.current_encoding = 0

        # MRD attributes
        self.measuremenInformation = []
        self.experimentalConditions = []
        self.encoding = []
        self.sequenceParameters = []
        self.userParameters = []
        self.waveformInformations = []

        # Arrays
        self.data = []  # Raw acquisitions
        self.waveforms = []  # Waveforms

        self.freeWaveformID = 1024

        self.label_dict = {
            "kspace_encoding_step_0": None,
            "kspace_encoding_step_1": None,
            "kspace_encoding_step_2": None,
            "average": None,
            "slice": None,
            "contrast": None,
            "phase": None,
            "repetition": None,
            "segment": None,
        }

    def mode_switch(func):
        def wrapper(self, *args, **kwargs):
            if self.mode != "rt" or self.mode != "static":
                return None
            return func(self, *args, **kwargs)

        return wrapper

    @mode_switch
    def calc_trajectory(self, *blocks: tuple[SimpleNamespace]):
        """
        Calculate readout header based on input PyPulseq blocks

        Parameters
        ----------
        *blocks : tuple[SimpleNamespace]
            List of blocks required to get readout.
            Examples:
                * line readout (cartesian): prewind + readout -> calc_trajectory((gx_pre,), (gx, adc))
                * spiral readout: readout only -> calc_trajectory((spiral_x_grad, spiral_y_grad, adc))

        Notes
        -----
        We assume that ADC is designed to cover only the actual readout part
        of the gradient, i.e., discard_pre=discard_post=0.

        """
        # build sequence based on provided blocks
        seq = pp.Sequence(system=DUMMY_SYSTEM)
        for block in blocks:
            seq.add_block(*block)

        # calculate k space
        k_traj_adc, _, _, _, t_adc = seq.calculate_kspace()

        # calculate sample time
        sample_time_us = np.unique(np.round(np.diff(t_adc), 12)).item() * 1e6

        # calculate number of samples
        number_of_samples = k_traj_adc.shape[-1]

        # calculate center sample
        k_space_zeros = np.all(k_traj_adc == 0, axis=0)  # True where column = (0,0,0)

        if np.any(k_space_zeros):
            center_sample = np.argmax(k_space_zeros)  # first True
        else:
            distances = np.linalg.norm(k_traj_adc, axis=0)  # Euclidean norm per column
            idx_last = np.argmax(-np.round(distances, 12)[::-1])
            center_sample = len(distances) - 1 - idx_last

        # get non uniform trajectory
        traj = []
        trajectory_dimensions = 3

        dkz = np.unique(np.round(np.diff(k_traj_adc[2]), 12))
        if dkz.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[2])

        dky = np.unique(np.round(np.diff(k_traj_adc[1]), 12))
        if dky.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[1])

        dkx = np.unique(np.round(np.diff(k_traj_adc[0]), 12))
        if dkx.size == 1:
            trajectory_dimensions -= 1
        else:
            traj.append(k_traj_adc[0])

        traj = traj[::-1]
        if traj:
            traj = np.stack(traj, axis=1)

        return SimpleNamespace(
            sample_time_us=int(sample_time_us),
            number_of_samples=number_of_samples,
            center_sample=center_sample,
            traj=traj,
            trajectory_dimensions=trajectory_dimensions,
        )

    @mode_switch
    def new_encoding(self):
        self.current_encoding = len(self.encodings) - 1

        exp = xsd.experimentalConditionsType()
        self.experimentalConditions.append(exp)

        meas = xsd.measurementInformationType()
        self.measuremenInformation.append(meas)

        encoding = xsd.encodingType()
        encoding.encodedSpace = xsd.encodingSpaceType()
        encoding.encodedSpace.matrixSize = xsd.matrixSizeType()
        encoding.encodedSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.reconSpace = xsd.encodingSpaceType()
        encoding.reconSpace.matrixSize = xsd.matrixSizeType()
        encoding.reconSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.encodingLimits = xsd.encodingLimitsType()
        encoding.echoTrainLength = 1
        encoding.trajectory = xsd.trajectoryType.CARTESIAN  # default
        encoding.trajectoryDescription = xsd.trajectoryDescriptionType()
        encoding.parallelImaging = xsd.parallelImagingType()

        self.encodings.append(encoding)

        seqParams = xsd.sequenceParametersType()
        self.sequenceParameters.append(seqParams)

        userParams = xsd.userParametersType()
        self.userParameters.append(userParams)

    @mode_switch
    def set_encoding(self, idx):
        if idx < 0:
            raise IndexError("Encoding index out of range")
        if idx >= self.current_encoding:
            while idx < len(self.encodings):
                self.new_encoding()
        self.current_encoding = idx

    def set_name(self, value):
        self.measuremenInformation[self.current_encoding].sequenceName = value

    @mode_switch
    def set_H1resonanceFrequency_Hz(self, gamma: float, B0: float):
        self.experimentalConditions[self.current_encoding].H1resonanceFrequency_Hz = (
            gamma * B0
        )

    @mode_switch
    def set_fov(self, dx: float, dy: float, dz: float):
        self.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.x = dx
        self.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.y = dy
        self.encoding[self.current_encoding].encodedSpace.fieldOfView_mm.z = dz
        self.encoding[self.current_encoding].reconSpace.fieldOfView_mm.x = dx
        self.encoding[self.current_encoding].reconSpace.fieldOfView_mm.y = dy
        self.encoding[self.current_encoding].reconSpace.fieldOfView_mm.z = dz

    @mode_switch
    def set_matrix(self, nx: int, ny: int, nz: int):
        self.encoding[self.current_encoding].encodedSpace.matrixSize.x = nx
        self.encoding[self.current_encoding].encodedSpace.matrixSize.y = ny
        self.encoding[self.current_encoding].encodedSpace.matrixSize.z = nz
        self.encoding[self.current_encoding].reconSpace.matrixSize.x = nx
        self.encoding[self.current_encoding].reconSpace.matrixSize.y = ny
        self.encoding[self.current_encoding].reconSpace.matrixSize.z = nz

    @mode_switch
    def set_limits(
        self, axis: str, maximum: int, minimum: int = 0, center: int | None = None
    ):
        key = axis2key(axis)
        limit = getattr(self.encoding[self.current_encoding].encodingLimits, key)
        limit.minimum = minimum
        limit.maximum = maximum
        if center is None:
            center = int(maximum // 2)
        limit.center = center
        setattr(self.encoding[self.current_encoding].encodingLimits, key, limit)

    @mode_switch
    def set_etl(self, value: int):
        self.encoding[self.current_encoding].echoTrainLength = value

    @mode_switch
    def set_trajectory(
        self,
        traj_type: xsd.trajectoryType = OTHER,
        desc: dict | None = None,
        comment: str = "",
    ):
        # Set trajectory type
        if isinstance(traj_type, Enum) is False:
            raise ValueError("traj_type must be an Enum")
        if traj_type in xsd.trajectoryType is False:
            raise ValueError("traj_type must be a valid trajectoryType")
        self.encoding[self.current_encoding].trajectory = traj_type

        # Set trajectory description
        if desc is None:
            desc = {}

        # fill appropriate fields
        udouble = []
        ulong = []
        ustring = []
        for k, v in desc:
            if isinstance(v, float):
                el = xsd.userParameterDoubleType(name=k, value=v)
                udouble.append(el)
            if isinstance(v, int):
                el = xsd.userParameterLongType(name=k, value=v)
                ulong.append(el)
            if isinstance(v, str):
                el = xsd.userParameterStringType(name=k, value=v)
                ustring.append(el)
        self.encoding[
            self.current_encoding
        ].trajectoryDescription.userParameterDouble = udouble
        self.encoding[self.current_encoding].trajectoryDescription.userParameterLong = (
            ulong
        )
        self.encoding[
            self.current_encoding
        ].trajectoryDescription.userParameterString = ustring

        # Set trajectory comment
        self.encoding[self.current_encoding].comment = comment

    @mode_switch
    def set_parallel_imaging_info(
        self,
        calibration_type: xsd.calibrationModeType,
        Ry: int = 1,
        Rz: int = 1,
        interleaving_dim: xsd.interleavingDimensionType | None = None,
    ):
        """
        Set parallel imaging header info.

        Parameters
        ----------
        calibration_type : xsd.calibrationModeType
            Type of Parallel Imaging calibration (e.g., EXTERNAL, EMBEDDED).
        Ry : int, optional
            Acceleration along ``y``(phase encoding dir).
            The default is ``1`` (no acceleration).
        Rz : int, optional
            Acceleration along ``z``(partition encoding dir).
            The default is ``1`` (no acceleration).

        """
        if calibration_type in xsd.calibrationModeType is False:
            raise ValueError("calibration_type must be a valid calibrationModeType")
        parallelImaging = xsd.parallelImagingType()
        parallelImaging.accelerationFactor.kspace_encoding_step_1 = Ry
        parallelImaging.accelerationFactor.kspace_encoding_step_2 = Rz
        parallelImaging.calibrationMode = calibration_type
        parallelImaging.interleavingDimension = interleaving_dim
        self.encoding[self.current_encoding].parallelImaging = parallelImaging

    @mode_switch
    def set_multiband_info(
        self,
        calibration_type: xsd.multibandCalibrationType,
        mb_factor: int,
        spacing: float | list[float],
        calibration_encoding: int,
        kz_shift: float = 0.0,
    ):
        """
        Set multiband header info.

        Parameters
        ----------
        calibration_type : xsd.multibandCalibrationType
            Type of Multiband calibration (e.g., FULL_3D).
        mb_factor : int
            Multiband acceleration factor.
        spacing : float | list[float]
            Spacing between slices in a single multiband packet.
        calibration_encoding : int
            Index of reference encoding space containing calibration data.
        kz_shift : float, optional
            Caipirinha shift between slizes.
            The default is ``0.0`` (No CAIPI shift).

        """
        if calibration_type in xsd.multibandCalibrationType is False:
            raise ValueError(
                "calibration_type must be a valid multibandCalibrationType"
            )

        if isinstance(spacing, float):
            spacing = [spacing]
        multiband_spacing = [
            xsd.multibandSpacingType((np.arange(mb_factor) * dz).tolist())
            for dz in spacing
        ]

        multiband = xsd.multibandType()
        multiband.spacing = multiband_spacing
        multiband.deltaKz = kz_shift
        multiband.calibration = calibration_type
        multiband.calibration_encoding = calibration_encoding

        if self.encoding[self.current_encoding].parallelImaging is None:
            self.encoding[self.current_encoding].parallelImaging = (
                xsd.parallelImagingType()
            )
        self.encoding[self.current_encoding].parallelImaging.multiband = multiband

    @mode_switch
    def set_TR(self, value):
        self.sequenceParams[self.current_encoding].TR.append(value)

    @mode_switch
    def set_TE(self, value):
        self.sequenceParams[self.current_encoding].TE.append(value)

    @mode_switch
    def set_TI(self, value):
        self.sequenceParams[self.current_encoding].TI.append(value)

    @mode_switch
    def set_flipAngle_deg(self, value):
        self.sequenceParams[self.current_encoding].flipAngle_deg.append(value)

    @mode_switch
    def set_sequence_type(self, value):
        self.sequenceParams[self.current_encoding].sequence_type = value

    @mode_switch
    def set_echo_spacing(self, value):
        self.sequenceParams[self.current_encoding].echo_spacing.append(value)

    @mode_switch
    def set_diffusion(
        self,
        channel: xsd.diffusionDimensionType,
        scheme: str,
        direction: np.ndarray,
        bvalue: float | np.ndarray,
    ):
        """
        Set diffusion parameters for current encoding.

        Parameters
        ----------
        channel : xsd.diffusionDimensionType
            Data axis representing diffusion direction (e.g., SEGMENT).
        scheme : str
            Diffusion encoding scheme (e.g., bipolar).
        direction : np.ndarray
            Array of shape ``(n, 3)`` representing diffusion directions.
        bvalue : float | np.ndarray
            Diffusion b-value or array of b-values of shape ``(n,)``.

        """
        if channel in xsd.diffusionDimensionType is False:
            raise ValueError("channel must be a valid diffusionDimensionType")

        direction = np.atleast_2d(direction)
        bvalue = np.atleast_1d(bvalue)
        if bvalue.shape[0] != 1 and bvalue.shape[0] != direction.shape[0]:
            raise ValueError(
                "Number of bvalues must match number of diffusion directions"
            )
        if bvalue.shape[0] == 1:
            bvalue = np.repeat(bvalue, direction.shape[0])

        gradDir = [
            xsd.gradientDirectionType(rl=el[0], ap=el[1], fh=el[2]) for el in direction
        ]
        diffusionParams = [
            xsd.diffusionType(gradDir[n], bvalue[n]) for n in range(bvalue.shape[0])
        ]

        self.sequenceParameters[self.current_encoding].diffusionDimension = channel
        self.sequenceParameters[self.current_encoding].diffusionScheme = scheme
        self.sequenceParameters[self.current_encoding].diffusion = diffusionParams

    @mode_switch
    def add_user_param(self, name, value):
        if isinstance(value, float):
            if haskey(
                name, self.userParameters[self.current_encoding].userParameterDouble
            ):
                setparam(
                    name,
                    value,
                    self.userParameters[self.current_encoding].userParameterDouble,
                )
            else:
                el = xsd.userParameterDoubleType(name=name, value=value)
                self.userParameters[self.current_encoding].userParameterDouble.append(
                    el
                )
                return
        if isinstance(value, int):
            if haskey(
                name, self.userParameters[self.current_encoding].userParameterLong
            ):
                setparam(
                    name,
                    value,
                    self.userParameters[self.current_encoding].userParameterLong,
                )
            else:
                el = xsd.userParameterLongType(name=name, value=value)
                self.userParameters[self.current_encoding].userParameterLong.append(el)
                return
        if isinstance(value, str):
            if haskey(
                name, self.userParameters[self.current_encoding].userParameterString
            ):
                setparam(
                    name,
                    value,
                    self.userParameters[self.current_encoding].userParameterString,
                )
            else:
                el = xsd.userParameterStringType(name=name, value=value)
                self.userParameters[self.current_encoding].userParameterString.append(
                    el
                )
                return

        # if not scalar, default to waveforms
        self._add_to_waveform(name, value)

    def _add_to_waveform(self, name, value):
        value = np.atleast_2d(value)
        found = False
        for el in self.waveformInformations:
            if el.waveformName == name:
                found = True
                waveId = el.waveformType
        if not (found):
            waveId = self.freeWaveformID
            newId = xsd.waveformInformationType(waveformName=name, waveformType=waveId)
            self.waveformInformations.append(newId)
            self.freeWaveformID += 1

        numChannels = value.shape[0]
        numSamples = value.shape[1]
        newWaveformHeader = mrd.WaveformHeader(
            waveform_id=waveId, channels=numChannels, number_of_samples=numSamples
        )
        newWaveform = mrd.Waveform(head=newWaveformHeader, data=value)
        self.waveforms.append(newWaveform)

    @mode_switch
    def add_acquisition(
        self,
        trajectory: SimpleNamespace,
        *events: SimpleNamespace,
        flag: int | None = None,
        encoding_idx: int | None = None,
        user_int: list[int] | None = None,
        user_float: list[float] | None = None,
    ):

        acq = mrd.Acquisition()

        # set flags
        if flag is not None:
            acq.setFlag(flag)

        # set scan counter
        acq.scan_counter = len(self.acquisitions)

        # resize trajectory
        acq.resize(
            trajectory_dimensions=trajectory.trajectory_dimensions,
            number_of_samples=trajectory.number_of_samples,
        )

        # set center sample
        acq.center_sample = trajectory.center_sample

        # set encoding space index
        if encoding_idx is None:
            encoding_idx = self.current_encoding
        acq.encoding_space_ref = encoding_idx

        # set sampling time
        acq.sample_time_us = trajectory.sample_time_us

        # set labels
        self.set_labels(*events)
        for k, v in self.label_dict.items():
            if v is not None:
                setattr(acq.idx, k, v)

        # set user defined parameters
        if user_int is not None:
            if len(user_int) > len(acq.user_int):
                warnings.warn(
                    "Length of provided user_int is larger than ismrmrd.Acquisition.user_int - ignoring extra entries",
                    stacklevel=2,
                )
            for n in range(len(acq.user_int)):
                acq.user_int = user_int[n]
        if user_float is not None:
            if len(user_float) > len(acq.user_float):
                warnings.warn(
                    "Length of provided user_float is larger than ismrmrd.Acquisition.user_float - ignoring extra entries",
                    stacklevel=2,
                )
            for n in range(len(acq.user_float)):
                acq.user_float = user_float[n]
                # acq.encoding

        # set trajectory
        rotation = None
        for event in events:
            if event.type == "rot3D":
                rotation = event.rot_quaternion
        if trajectory.traj:
            traj = copy.deepcopy(trajectory.traj)
            if rotation is not None:
                traj = rotation.apply(traj)
        else:
            traj = None
        if traj is not None:
            acq.traj[:] = traj

        self.acquisitions.append(acq)

    def set_labels(self, *events):
        for event in events:
            if event.type == "labelset":
                label = axis2key(event.label)
                if label in self.label_dict:
                    self.label_dict[label] = event.value
            if event.type == "labelinc":
                label = axis2key(event.label)
                if label in self.label_dict:
                    if self.label_dict[label] is None:
                        self.label_dict[label] = 0
                    self.label_dict[label] += event.value

    #######################
    # Retrieve header
    #######################

    def get_header(self, encoding_index=0):
        hdr = xsd.ismrmrdHeader()
        hdr.experimentalConditions = self.experimentalConditions

        # Copy encodings (encodedSpace, reconSpace, encodingLimits already per encoding)
        for i, enc in enumerate(self.encodings):
            enc_copy = xsd.encodingType()
            enc_copy.encodedSpace = enc.encodedSpace
            enc_copy.reconSpace = enc.reconSpace
            enc_copy.encodingLimits = enc.encodingLimits

            # Apply shared globals only to selected encoding
            if i == encoding_index:
                if self._trajectories[i]:
                    enc_copy.trajectory = self._trajectories[i]
                if self._echoTrainLengths[i] is not None:
                    enc_copy.echoTrainLength = self._echoTrainLengths[i]
                if self._parallelImagings[i]:
                    enc_copy.parallelImaging = self._parallelImagings[i]
            hdr.encoding.append(enc_copy)

        # Sequence parameters for selected encoding
        if encoding_index < len(self._sequenceParams):
            hdr.sequenceParameters = self._sequenceParams[encoding_index]

        # User parameters for selected encoding
        if encoding_index < len(self._userParams):
            hdr.userParameters = self._userParams[encoding_index]

        # Waveforms for selected encoding
        if encoding_index < len(self._waveforms):
            hdr.waveformInformation.extend(self._waveforms[encoding_index])

        return hdr


# %% utils
def axis2key(axis):
    if axis.lower() in _axis2key:
        return _axis2key[axis.lower()]
    else:
        return axis.lower()


_axis2key = {
    "k0": "kspace_encoding_step_0",
    "acq": "kspace_encoding_step_0",
    "k1": "kspace_encoding_step_1",
    "lin": "kspace_encoding_step_1",
    "k2": "kspace_encoding_step_2",
    "par": "kspace_encoding_step_2",
    "avg": "average",
    "slc": "slice",
    "eco": "contrast",
    "phs": "phase",
    "rep": "repetition",
    "seg": "segment",
    "user0": "user_0",
    "user1": "user_1",
    "user2": "user_2",
    "user3": "user_3",
    "user4": "user_4",
    "user5": "user_5",
    "user6": "user_6",
    "user7": "user_7",
}


def haskey(key, *userparams):
    for user in userparams:
        for el in user:
            if el.name == key:
                return True
    return False


def setparam(key, value, *userparams):
    for user in userparams:
        for el in user:
            if el.name == key:
                el.value = value
