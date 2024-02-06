"""
Class for reading data from WaveSurfer, a software written by
Boaz Mohar and Adam Taylor https://wavesurfer.janelia.org/

This is a wrapper around the PyWaveSurfer module written by Boaz Mohar and Adam Taylor,
using the "double" argument to load the data as 64-bit double.
"""
import numpy as np
import quantities as pq
import re

from neo.io.baseio import BaseIO
from neo.core import Block, Segment, AnalogSignal
from ..rawio.baserawio import _signal_channel_dtype, _signal_stream_dtype, _spike_channel_dtype, _event_channel_dtype  # TODO: not sure about this  # from ..rawio.

try:
    from pywavesurfer import ws
except ImportError as err:
    HAS_PYWAVESURFER = False
    PYWAVESURFER_ERR = err
else:
    HAS_PYWAVESURFER = True
    PYWAVESURFER_ERR = None


def get_ordered_list_of_sweeps(ws_rec_keys):
    """
    In some cases the sweep number in wavesurfer file do not start
    at zero/one. e.g. ws_rec.keys() == ["header", "sweep_0004", "sweep_0005"].
    In order to process these, we need to get their order. If this order is
    incorrect it would be very bad as traces will be presented in wrong order.

    Parse the "sweep_xxxx" dict keys and return them in order. Do some sanity
    checks to be 100% sure the returned keys are in order.
    """
    data_keys = list(ws_rec_keys)
    data_keys.remove("header")

    idx_order = np.argsort([int(sweep_name.split("_")[1]) for sweep_name in data_keys])
    sweeps_ordered = [data_keys[idx] for idx in idx_order]

    sanity_check_order = [int(re.search(r'\d+', sweep).group()) for sweep in sweeps_ordered]

    assert sorted(sanity_check_order) == sanity_check_order
    assert len(sweeps_ordered) == len(data_keys)

    return sweeps_ordered

class WaveSurferIO(BaseIO):
    """
    """

    is_readable = True
    is_writable = False

    supported_objects = [Block, Segment, AnalogSignal]
    readable_objects = [Block]
    writeable_objects = []

    has_header = True
    is_streameable = False

    read_params = {Block: []}
    write_params = None

    name = 'WaveSurfer'
    extensions = ['.h5']

    mode = 'file'

    def __init__(self, filename=None):
        """
        Arguments:
            filename : a filename
        """
        if not HAS_PYWAVESURFER:
            raise PYWAVESURFER_ERR

        BaseIO.__init__(self)

        self.filename = filename
        self.ws_rec = None
        self.header = {}
        self._sampling_rate = None
        self.ai_channel_names = None
        self.ai_channel_units = None

        self.read_block()

    def read_block(self, lazy=False):
        assert not lazy, 'Do not support lazy'

        self.ws_rec = ws.loadDataFile(self.filename, format_string="double")

        ai_channel_names = self.ws_rec["header"]["AIChannelNames"].astype(str)
        ai_channel_units = self.ws_rec["header"]["AIChannelUnits"].astype(str)
        sampling_rate = np.float64(self.ws_rec["header"]["AcquisitionSampleRate"]) * 1 / pq.s

        self.fill_header(ai_channel_names,
                         ai_channel_units)

        bl = Block()

        sweeps_ordered = get_ordered_list_of_sweeps(self.ws_rec.keys())

        # iterate over sections first:
        for seg_index, sweep_key in enumerate(sweeps_ordered):

            seg = Segment(index=seg_index)
            # seg_id = "sweep_{0:04d}".format(seg_index + 1)  # e.g. "sweep_0050"

            ws_seg = self.ws_rec[sweep_key]

            t_start = np.float64(ws_seg["timestamp"]) * pq.s

            # iterate over channels:
            for chan_idx, recsig in enumerate(ws_seg["analogScans"]):

                unit = ai_channel_units[chan_idx]
                name = ai_channel_names[chan_idx]

                signal = pq.Quantity(recsig, unit).T

                anaSig = AnalogSignal(signal, sampling_rate=sampling_rate,
                                      t_start=t_start, name=name,
                                      channel_index=chan_idx)
                seg.analogsignals.append(anaSig)
            bl.segments.append(seg)

        # bl.create_many_to_one_relationship()  removed in version 13.

        return bl

    def fill_header(self, ai_channel_names, ai_channel_units):

        signal_channels = []

        for ch_idx, (ch_name, ch_units) in enumerate(zip(ai_channel_names,
                                                         ai_channel_units)):
            ch_id = ch_idx + 1
            dtype = "float64"  # as loaded with "double" argument from PyWaveSurfer
            gain = 1
            offset = 0
            stream_id = "0"
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
        self.header['nb_block'] = 1
        self.header['nb_segment'] = [int(self.ws_rec["header"]["NSweepsPerRun"])]
        self.header['signal_streams'] = signal_streams
        self.header['signal_channels'] = signal_channels
        self.header['spike_channels'] = spike_channels
        self.header['event_channels'] = event_channels
