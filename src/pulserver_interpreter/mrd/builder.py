"""MRD builder."""

__all__ = ["ISMRMRDBuilder"]

import ismrmrd.xsd as xsd

class ISMRMRDBuilder:
    def __init__(self, mode='dry'):
        self.mode = mode  # 'dry', 'prep', 'eval', 'rt'
        self.current_encoding = 0

        # MRD attributes
        self.experimentalConditions = []
        self.encoding = []  
        self.sequenceParameters = []
        self.userParameters = []
        self.waveformInformations = []
        
        # Arrays
        self.data = [] # Raw acquisitions
        self.waveforms = [] # Waveforms
        
    #######################
    # Realtime decorator
    #######################
    def rt_only(func):
        def wrapper(self, *args, **kwargs):
            if self.mode != 'rt':
                return self
            return func(self, *args, **kwargs)
        return wrapper

    #######################
    # Encoding management
    #######################
    @rt_only
    def new_encoding(self):
        self.current_encoding = len(self.encodings) - 1
        
        exp = xsd.experimentalConditionsType()
        self.experimentalConditions.append(exp)

        encoding = xsd.encodingType()
        encoding.encodedSpace = xsd.encodingSpaceType()
        encoding.encodedSpace.matrixSize = xsd.matrixSizeType()
        encoding.encodedSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.reconSpace = xsd.encodingSpaceType()
        encoding.reconSpace.matrixSize = xsd.matrixSizeType()
        encoding.reconSpace.fieldOfView_mm = xsd.fieldOfViewMm()
        encoding.encodingLimits = xsd.encodingLimitsType()
        encoding.trajectory = xsd.trajectoryType.CARTESIAN # default
        # encoding.trajectoryDescription = xsd.trajectoryDescriptionType()
        encoding.parallelImaging = xsd.parallelImagingType()

        self.encodings.append(encoding)
        
    
        

    @rt_only
    def set_encoding(self, idx):
        if idx < 0:
            raise IndexError('Encoding index out of range')
        if idx >= self.current_encoding:
            self.new_encoding()
        self.current_encoding = idx
        return self

    #######################
    # Shared globals setters
    #######################

    @rt_only
    def set_trajectory(self, traj):
        self._trajectories[self.current_encoding] = xsd.trajectoryType(traj)
        return self

    @_rt_only
    def set_echoTrainLength(self, val):
        self._echoTrainLengths[self.current_encoding] = val
        return self

    @_rt_only
    def set_parallelImaging(self, accelerationFactor=None, calibrationMode=None,
                            interleavingDimension=None, multiband=None):
        pi = self._parallelImagings[self.current_encoding]
        if accelerationFactor: pi.accelerationFactor = accelerationFactor
        if calibrationMode: pi.calibrationMode = xsd.calibrationModeType(calibrationMode)
        if interleavingDimension: pi.interleavingDimension = xsd.interleavingDimensionType(interleavingDimension)
        if multiband: pi.multiband = multiband
        return self

    #######################
    # Sequence parameters
    #######################

    @_rt_only
    def set_TR(self, val):
        self._sequenceParams[self.current_encoding].TR.append(val)
        return self

    @_rt_only
    def set_TE(self, val):
        self._sequenceParams[self.current_encoding].TE.append(val)
        return self

    @_rt_only
    def set_TI(self, val):
        self._sequenceParams[self.current_encoding].TI.append(val)
        return self

    @_rt_only
    def set_flipAngle_deg(self, val):
        self._sequenceParams[self.current_encoding].flipAngle_deg.append(val)
        return self

    @_rt_only
    def set_sequence_type(self, val):
        self._sequenceParams[self.current_encoding].sequence_type = val
        return self

    @_rt_only
    def set_echo_spacing(self, val):
        self._sequenceParams[self.current_encoding].echo_spacing.append(val)
        return self

    @_rt_only
    def set_diffusionDimension(self, val):
        self._sequenceParams[self.current_encoding].diffusionDimension = xsd.diffusionDimensionType(val)
        return self

    @_rt_only
    def add_diffusion(self, gradientDirection: dict, bvalue: float):
        diff = xsd.diffusionType()
        diff.gradientDirection = xsd.gradientDirectionType()
        diff.gradientDirection.rl = gradientDirection.get('rl', 0.0)
        diff.gradientDirection.ap = gradientDirection.get('ap', 0.0)
        diff.gradientDirection.fh = gradientDirection.get('fh', 0.0)
        diff.bvalue = bvalue
        self._sequenceParams[self.current_encoding].diffusion.append(diff)
        return self

    @_rt_only
    def set_diffusionScheme(self, val):
        self._sequenceParams[self.current_encoding].diffusionScheme = val
        return self

    #######################
    # User parameters
    #######################

    @_rt_only
    def add_userParameterLong(self, name, value, array=False):
        if array:
            self._add_to_waveform(name, value, 'long')
        else:
            param = xsd.userParameterLongType()
            param.name = name
            param.value_ = value
            self._userParams[self.current_encoding].userParameterLong.append(param)
        return self

    @_rt_only
    def add_userParameterDouble(self, name, value, array=False):
        if array:
            self._add_to_waveform(name, value, 'double')
        else:
            param = xsd.userParameterDoubleType()
            param.name = name
            param.value_ = value
            self._userParams[self.current_encoding].userParameterDouble.append(param)
        return self

    @_rt_only
    def add_userParameterString(self, name, value, array=False):
        if array:
            self._add_to_waveform(name, value, 'string')
        else:
            param = xsd.userParameterStringType()
            param.name = name
            param.value_ = value
            self._userParams[self.current_encoding].userParameterString.append(param)
        return self

    def _add_to_waveform(self, name, value, dtype):
        # check if waveform exists for current encoding
        wf_list = self._waveforms[self.current_encoding]
        if not wf_list:
            wf = xsd.waveformInformationType()
            wf.waveformName = name
            wf.waveformType = dtype
            wf.userParameters = xsd.userParametersType()
            wf_list.append(wf)
        else:
            wf = wf_list[0]  # for simplicity, single waveform per encoding for now
        # append to appropriate type
        if dtype == 'long':
            param = xsd.userParameterLongType()
            param.name = name
            param.value_ = value
            wf.userParameters.userParameterLong.append(param)
        elif dtype == 'double':
            param = xsd.userParameterDoubleType()
            param.name = name
            param.value_ = value
            wf.userParameters.userParameterDouble.append(param)
        elif dtype == 'string':
            param = xsd.userParameterStringType()
            param.name = name
            param.value_ = value
            wf.userParameters.userParameterString.append(param)

    #######################
    # Experimental conditions
    #######################

    @_rt_only
    def set_H1resonanceFrequency_Hz(self, val: int):
        self.experimentalConditions.H1resonanceFrequency_Hz = val
        return self

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