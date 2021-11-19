"""
"""

from .baserawio import (BaseRawIO, _signal_channel_dtype, _signal_stream_dtype,
                        _spike_channel_dtype, _event_channel_dtype)
import numpy as np

try:
    from load_heka_python import LoadHeka  # TOOD: what is package called?
except ImportError as err:
    HAS_LOADHEKA = False
    LOADHEKA_ERR = err
else:
    HAS_LOADHEKA = True
    LOADHEKA_ERR = None

class HekaRawIO(BaseRawIO):

    extensions = ['.dat']
    rawmode = 'one-file'

    def __init__(self, filename=''):
        BaseRawIO.__init__(self)
        self.filename = filename

        if not HAS_LOADHEKA:
            raise LOADHEKA_ERR

    def _source_name(self):
        return self.filename

    def _parse_header(self):

        heka_data = LoadHeka(self.filename,
                             only_load_header=True)

        header = heka_data["header"]

        group_idx = 0
        series_idx = 0

        group = heka_data["ch"][group_idx]

        # Raw Data
        self._raw_signals = {}
        self._t_starts = {}

  #      for seg_index in range(int(header["NSweepsPerRun"])):
 #           sweep_id = "sweep_{0:04d}".format(seg_index + 1)                     # e.g. "sweep_0050"
  #          self._raw_signals[seg_index] = pyws_data[sweep_id]["analogScans"].T  # reshape to data x channel for Neo standard
   #         self._t_starts[seg_index] = np.float64(pyws_data[sweep_id]["timestamp"])

        # Signal Channels

        num_blocks = len(group["ch"])
        segments_per_block = [series["hd"]["SeNumberSweeps"] for series in group["ch"]]
        channels = heka.get_channels(group_idx)  # TODO: it is not clear for heka files that all blocks will have the same channel order. At least,they can have different number of channels per block. This will be tested in assert at least in load_heka
        # the sampling rate is not garenteed to be the same across blocks

        signal_channels = []
        for ch_idx, chan in enumerate(channels):
            ch_id = ch_idx + 1
            dtype = chan["dtype"]  # TODO: currently this is how the data are stored in HEKA raw. however HEKA loader will convert all to float64. Should this always be float64?
            name = chan["name"]
            gain = 1
            offset = 0

        signal_channels.append((chan["name"], ch_id, self._sampling_rate, dtype, ch_units, gain, offset, stream_id))

        get_series_channels

        # TODO: add getters to get this data (i.e. dont just print)
        # channel names
        # channel units
        # gain
        # offset  sVpOffset  sStimDacOffset  sVmonOffset    TrZeroData
        # stream_id
        # dtype
        # num sweeps

        signal_channels = []
        ai_channel_names = header["AIChannelNames"].astype(str)
        ai_channel_units = header["AIChannelUnits"].astype(str)
        self._sampling_rate = np.float64(pyws_data["header"]["AcquisitionSampleRate"])

        for ch_idx, (ch_name, ch_units) in enumerate(zip(ai_channel_names,
                                                         ai_channel_units)):
            ch_id = ch_idx + 1
            dtype = "float64"  # as loaded with "double" argument from PyWaveSurfer
            gain = 1
            offset = 0
            stream_id = "0"  # chan_id
            signal_channels.append((ch_name, ch_id, self._sampling_rate, dtype, ch_units, gain, offset, stream_id))

        signal_channels = np.array(signal_channels, dtype=_signal_channel_dtype)

        # Spike Channels (no spikes)
        spike_channels = []
        spike_channels = np.array(spike_channels, dtype=_spike_channel_dtype)

        # Event Channels (no events)
        event_channels = []
        event_channels = np.array(event_channels, dtype=_event_channel_dtype)

        # Signal Streams
        signal_streams = np.array([('Signals', '0')], dtype=_signal_stream_dtype)

        # Header Dict
        self.header = {}
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [int(header["NSweepsPerRun"])]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels

        self._generate_minimal_annotations()  # TODO: return to this and add annotations

    # TODO: check that the channels are all the same across series

    def _segment_t_start(self, block_index, seg_index):
        return self._t_starts[seg_index]

    def _segment_t_stop(self, block_index, seg_index):
        t_stop = self._t_starts[seg_index] + \
                 self._raw_signals[seg_index].shape[0] / self._sampling_rate
        return t_stop

    def _get_signal_size(self, block_index, seg_index, stream_index):
        shape = self._raw_signals[seg_index].shape
        return shape[0]

    def _get_signal_t_start(self, block_index, seg_index, stream_index):
        return self._t_starts[seg_index]

    def _get_analogsignal_chunk(self, block_index, seg_index, i_start, i_stop,
                                stream_index, channel_indexes):
        if channel_indexes is None:
            channel_indexes = slice(None)
        raw_signals = self._raw_signals[seg_index][slice(i_start, i_stop), channel_indexes]
        return raw_signals
