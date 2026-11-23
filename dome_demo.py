"""
Dome sound demo -- a looping sound scene in the freefield dome, with moving sounds.

Every sound is loaded once, whole, into its own looping buffer. Static sounds
play from one loudspeaker. Moving sounds follow a fixed path and are panned
across the loudspeakers with a Gaussian weight around their current direction,
as in Marc's old RX8_play_multichannel script. One zBus trigger starts all
buffers on both RX8s in the same sample. While the demo runs, each sound can be
muted or turned up/down from the console.

Circuits: rcx/scene_RX81.rcx and rcx/scene_RX82.rcx (one per RX8, RX8-2, 48828 Hz)
------------------------------------------------------------------------------
Trimmed from Marc's 24ch_8Sig.rcx (8 x 24 gains on 6 DSPs) to fit two 2-DSP RX8s:
only the moving sounds get a gain on every channel, static sounds get one gain
and a channel that is wired in the circuit.

    buffers 1..N_MOVING (both RX8s) -- moving sounds
      sigin{i}         SerSource buffer, Size = BUFFER_SIZE, loops
      len{i}           buffer i wraps back to 0 after len{i} samples
      aSig{i}_Ch{c}    gain of buffer i on channel c = 1..24, smoothed (as in Marc's circuit)
    buffers N_MOVING+1.. (per RX8, see STATIC_WIRING) -- static sounds
      sigin{i}, len{i} as above
      amp{i}           gain of buffer i, smoothed; output added to ONE channel, wired in the circuit
    per channel c      sum of everything routed to c -> one DacOut c
                       (one DacOut per channel: TDT does not document what two DacOuts
                       on the same channel do)
    zBusA (pulse)      reset all buffer indices to 0 and start (set an RSFlipFlop)
    zBusB (pulse)      stop (reset the RSFlipFlop)

STATIC_WIRING is the wiring of the static buffers; the script checks every static
sound against it. A static sound on a speaker that has no wired buffer can be made
a moving sound with a fixed position instead (path=lambda t: (az, el)).
Memory: buffers x BUFFER_SIZE floats per RX8 (logged at startup), of 128 MB.
"""
# %% imports
import logging
import math
import queue
import threading
import time
from collections import namedtuple
from pathlib import Path

import numpy
import slab
import freefield
# `import freefield` only exposes a copy of PROCESSORS (star import), so the
# module itself is needed to see the processors after initialize().
from freefield import freefield as freefield_core

logging.basicConfig(level=logging.INFO)


# %% trajectories -- functions of time t (s) returning (azimuth, elevation) in degrees
def ping_pong(start, end, period):
    """Back and forth between `start` and `end`, one round trip per `period`; slows at the ends."""
    def path(t):
        u = 0.5 - 0.5 * math.cos(2 * math.pi * t / period)
        return start[0] + u * (end[0] - start[0]), start[1] + u * (end[1] - start[1])
    return path


def orbit(center, radius, period):
    """Ellipse around `center` with radii (azimuth, elevation), one turn per `period`."""
    def path(t):
        phase = 2 * math.pi * t / period
        return center[0] + radius[0] * math.cos(phase), center[1] + radius[1] * math.sin(phase)
    return path


# ==================== quick config ====================
# Static sounds: wav file, (azimuth, elevation) of the dome speaker, level in dB.
# Moving sounds: wav file, a path from the functions above, level in dB.
# The level is the ACTIVE level: the RMS over the parts where the sound is on (100 ms
# windows less than ACTIVE_RANGE dB below the loudest one), not over the whole file --
# the cat is silent ~80 % of its file, so its plain RMS is ~7 dB below its meows.
# The cat is the target to localize: static, and small but above the ocean bed.
# These levels are a starting point -- balance them by ear in the dome with the
# console (e.g. '6 -2'), then copy the values the console prints back here.
SOUND_DIR = Path(__file__).resolve().parent / 'sounds'   # wav paths are relative to this
SOUNDS = [
    ('ocean.wav', (-52.5, -25), 58),
    ('cows.wav', (-35, -12.5), 62),
    ('chimes.wav', (-17.5, 25), 54),
    ('chimes2.wav', (0, 37.5), 52),
    ('chimes3.wav', (17.5, 25), 53),
    ('cat.wav', (17.5, -12.5), 52),                                      # target
    ('motor.wav', (35, -25), 54),
    ('geese.wav', ping_pong((-52.5, 30), (52.5, 30), period=24), 57),    # fly over, left <-> right
    ('birds.wav', orbit((0, 25), (40, 10), period=30), 56),              # circle up high
    ('train_s.wav', ping_pong((-52.5, 0), (52.5, 0), period=38), 53),    # pass along the horizon
    # ('train.wav', ...) -- 184 s: a buffer that long does not fit 8 times into the RX8
    #                       memory; train_s.wav is Marc's trimmed version of it.
]
WIDTH = 30              # deg, FWHM of the Gaussian that pans a moving sound (Marc's was ~90)
UPDATE_RATE = 10        # Hz, how often the gains of moving sounds are rewritten
MASTER_GAIN = 0         # dB added to every level -- turns the whole scene up or down
ACTIVE_RANGE = 30       # dB below the loudest 100 ms window that still counts as 'on'
LOOP_RAMP = 0.005       # s, raised-cosine ramp at both ends of each sound, so the wrap does not click
# ======================================================

# Fixed in the circuit -- these must match it (see the docstring):
SAMPLERATE = 48828      # the wavs are 22.05 kHz (content up to 11 kHz); at 24414 Hz the
                        # RX8 DAC filter would already roll off near 11 kHz
N_MOVING = 3            # buffers 1..3 on both RX8s: sigin, len, aSig{i}_Ch1..24
N_CHANNELS = 24         # aSig{i}_Ch1..24
STATIC_WIRING = {       # static buffer -> the DAC channel it is wired to in that RX8's circuit
    'RX81': {4: 11, 5: 4, 6: 20, 7: 8, 8: 21},   # chimes, chimes2, chimes3, cat, motor
    'RX82': {4: 18, 5: 2},                         # ocean, cows
}
BUFFER_SIZE = 2100000   # SerSource Size; cows.wav (42 s) is the longest sound: 2.05 M at 48828 Hz
RCX_FILES = {'RX81': 'scene_RX81.rcx', 'RX82': 'scene_RX82.rcx'}
RCX_DIR = Path(__file__).resolve().parent / 'rcx'
PROCESSOR_MODE = 'dome_demo'
PROCS = ('RX81', 'RX82')
MIN_GAIN = 0.01         # pan weights below this (-40 dB) are written as 0
MIN_CHANGE = 0.002      # a gain is only rewritten when it changed by more than this


# %% helpers
def _circuit(proc):
    """Path of the circuit for `proc`, in ./rcx."""
    path = RCX_DIR / RCX_FILES[proc]
    if not path.exists():
        raise FileNotFoundError(f'Could not find the {proc} circuit {path}.')
    return str(path)


Speaker = namedtuple('Speaker', 'index analog_channel analog_proc azimuth elevation')


def _dome_table():
    """The dome speaker table, read straight from freefield's csv by column name.

    freefield's own read_speaker_table() depends on the module-level SETUP, which
    is a string in the pinned version and a Setup object in newer ones.
    """
    file = Path(freefield.DIR) / 'data' / 'tables' / 'speakertable_dome.txt'
    rows = numpy.loadtxt(file, delimiter=',', dtype=str)
    header, rows = list(rows[0]), rows[1:]
    col = {name: header.index(name) for name in ('index_number', 'channel', 'analog_proc', 'azi', 'ele')}
    return [Speaker(int(r[col['index_number']]), int(r[col['channel']]), r[col['analog_proc']],
                    float(r[col['azi']]), float(r[col['ele']])) for r in rows]


def _find_speaker(table, azimuth, elevation):
    matches = [s for s in table if s.azimuth == azimuth and s.elevation == elevation]
    if not matches:
        raise ValueError(f'No dome speaker at azimuth {azimuth}, elevation {elevation}.')
    return matches[0]


def _unit_vectors(azimuth, elevation):
    """Unit vectors for (azimuth, elevation) in degrees."""
    az, el = numpy.radians(azimuth), numpy.radians(elevation)
    return numpy.stack([numpy.cos(el) * numpy.cos(az), numpy.cos(el) * numpy.sin(az),
                        numpy.sin(el)], axis=-1)


def pan_weights(azimuth, elevation, speaker_vectors, width=WIDTH):
    """Gaussian weights over the speakers for a source at (azimuth, elevation).

    The distance is the great-circle angle (deg) between source and speaker; the
    Gaussian has a FWHM of `width` deg. The weights are normalized to unit POWER
    (sum of squares = 1), so the level stays the same however many speakers share
    the sound. (Marc normalized the plain sum to 1, which makes a wide source
    quieter at high frequencies, where the speakers add up incoherently.)
    """
    cos_angle = numpy.clip(speaker_vectors @ _unit_vectors(azimuth, elevation), -1, 1)
    angle = numpy.degrees(numpy.arccos(cos_angle))
    sigma = width / (2 * math.sqrt(2 * math.log(2)))
    weights = numpy.exp(-angle ** 2 / (2 * sigma ** 2))
    weights /= numpy.sqrt((weights ** 2).sum())
    weights[weights < MIN_GAIN] = 0
    return weights


def _wav_path(file):
    file = Path(file)
    return file if file.is_absolute() else SOUND_DIR / file


def _active_level(sound, window=0.1):
    """Level (slab dB) of the windows where the sound is on, ignoring pauses and silence."""
    n_win = max(1, int(window * sound.samplerate))
    data = sound.data[:, 0]
    frames = data[:len(data) // n_win * n_win].reshape(-1, n_win)
    power = (frames ** 2).mean(axis=1)
    on = power >= power.max() * 10 ** (-ACTIVE_RANGE / 10)
    active_rms = numpy.sqrt(power[on].mean())
    return sound.level + 20 * numpy.log10(active_rms / numpy.sqrt((data ** 2).mean()))


def _load_sound(file, level):
    """Load a whole wav as a mono sound at the TDT rate, with its active level set to `level`."""
    sound = slab.Sound(_wav_path(file))
    if sound.n_channels > 1:  # freefield flattens 2-D arrays, which would interleave the channels
        sound = slab.Sound(sound.data.mean(axis=1), samplerate=sound.samplerate)
    if sound.samplerate != SAMPLERATE:
        sound = sound.resample(SAMPLERATE)
    sound.level = level + MASTER_GAIN + sound.level - _active_level(sound)
    if LOOP_RAMP:
        sound = sound.ramp(duration=LOOP_RAMP)
    return sound


# %% demo
class DomeDemo:
    """Load the sounds in SOUNDS onto the RX8s, loop them, and move the moving ones."""

    def __init__(self, sounds=SOUNDS):
        self.table = _dome_table()
        self.speaker_vectors = _unit_vectors(numpy.array([s.azimuth for s in self.table]),
                                             numpy.array([s.elevation for s in self.table]))
        self.sounds = []  # one dict per sound, in the order of SOUNDS
        for file, where, level in sounds:
            entry = {'name': Path(file).stem, 'level': level, 'offset': 0.0, 'muted': False,
                     'sound': _load_sound(file, level)}
            if callable(where):
                entry.update(moving=True, path=where, procs=PROCS)
            else:
                speaker = _find_speaker(self.table, *where)
                entry.update(moving=False, speaker=speaker, procs=(speaker.analog_proc,))
            entry['data'] = numpy.asarray(entry['sound'].data, dtype=float).flatten()
            entry['peak'] = numpy.abs(entry['data']).max()
            if len(entry['data']) > BUFFER_SIZE:
                raise ValueError(f"{entry['name']} is {len(entry['data'])} samples, the buffers hold "
                                 f'{BUFFER_SIZE}. Raise BUFFER_SIZE and the SerialBuf Size.')
            if entry['peak'] >= 1:  # +-1 is +-10 V on the RX8 DAC
                logging.warning('%s peaks at %.2f (>= 1): lower its level or it will clip.',
                                entry['name'], entry['peak'])
            self.sounds.append(entry)
        self._check_positions()
        self._assign_buffers()
        self._written = {}      # (proc, buffer, channel) -> last gain written
        self._t0 = None
        for proc in PROCS:
            n = N_MOVING + len(STATIC_WIRING[proc])
            logging.info('%s: %d buffers x %d samples = %.0f MB at %d Hz.',
                         proc, n, BUFFER_SIZE, n * BUFFER_SIZE * 4 / 1e6, SAMPLERATE)

    def _check_positions(self):
        static = [s['speaker'].index for s in self.sounds if not s['moving']]
        if len(static) != len(set(static)):
            raise ValueError('Two static sounds share a speaker.')

    def _assign_buffers(self):
        """Moving sounds take buffers 1..N_MOVING on both RX8s; a static sound takes the
        buffer that STATIC_WIRING connects to its speaker's channel."""
        moving = [s for s in self.sounds if s['moving']]
        if len(moving) > N_MOVING:
            raise ValueError(f'{len(moving)} moving sounds, the circuits have {N_MOVING} moving buffers.')
        for i, entry in enumerate(moving, start=1):
            entry['buffer'] = i
        for entry in self.sounds:
            if entry['moving']:
                continue
            speaker = entry['speaker']
            wired = {c: i for i, c in STATIC_WIRING[speaker.analog_proc].items()}
            if speaker.analog_channel not in wired:
                raise ValueError(
                    f"{entry['name']}: no static buffer on {speaker.analog_proc} is wired to channel "
                    f'{speaker.analog_channel} (speaker {speaker.index}). Wired: '
                    f'{STATIC_WIRING[speaker.analog_proc]}. Rewire the circuit or make it a moving '
                    'sound with a fixed position.')
            entry['buffer'] = wired[speaker.analog_channel]

    # ------------------------------------------------------------ processors
    def init_processors(self):
        if freefield_core.PROCESSORS is None or freefield_core.PROCESSORS.mode != PROCESSOR_MODE:
            freefield.initialize('dome', device=[[proc, 'RX8', _circuit(proc)] for proc in PROCS],
                                 zbus=True, sensor_tracking=False)
            freefield_core.PROCESSORS.mode = PROCESSOR_MODE
        for proc in PROCS:  # the rate is fixed by the .rcx; a wrong one plays at the wrong pitch
            rate = freefield_core.PROCESSORS.processors[proc].GetSFreq()
            if abs(rate - SAMPLERATE) > 1:
                raise RuntimeError(f'{proc} runs at {rate:.0f} Hz but SAMPLERATE is {SAMPLERATE}. '
                                   'Compile the circuit at that rate.')

    def load(self):
        """Write all buffers, lengths and initial gains. Call while the demo is stopped."""
        self.stop()
        for proc in PROCS:
            for entry in self.sounds:
                if proc in entry['procs']:
                    i = entry['buffer']
                    freefield.write(f'sigin{i}', entry['data'], proc)
                    freefield.write(f'len{i}', len(entry['data']), proc)
        self._written.clear()
        self.update_gains(t=0.0, force=True)
        for entry in self.sounds:
            where = ('moving' if entry['moving'] else
                     f"speaker {entry['speaker'].index:2d} (az {entry['speaker'].azimuth:g}, "
                     f"el {entry['speaker'].elevation:g})")
            logging.info('%-8s buffer %d on %s | %6.2f s | %d dB | %s', entry['name'],
                         entry['buffer'], '+'.join(entry['procs']),
                         len(entry['data']) / SAMPLERATE, entry['level'], where)

    def start(self):
        freefield.play(kind='zBusA')  # both RX8s: indices to 0 and run, in the same sample
        self._t0 = time.monotonic()

    def stop(self):
        freefield.play(kind='zBusB')
        self._t0 = None

    # ------------------------------------------------------------ gains
    def _target_gains(self, t):
        """{(proc, tag): gain} for every gain tag in the circuits, at scene time t."""
        gains = {(proc, f'aSig{i}_Ch{c}'): 0.0 for proc in PROCS
                 for i in range(1, N_MOVING + 1) for c in range(1, N_CHANNELS + 1)}
        gains.update({(proc, f'amp{i}'): 0.0 for proc in PROCS for i in STATIC_WIRING[proc]})
        for entry in self.sounds:
            scale = 0.0 if entry['muted'] else 10 ** (entry['offset'] / 20)
            if entry['moving']:
                weights = pan_weights(*entry['path'](t), self.speaker_vectors) * scale
                for speaker, w in zip(self.table, weights):
                    gains[(speaker.analog_proc, f"aSig{entry['buffer']}_Ch{speaker.analog_channel}")] = w
            else:
                gains[(entry['speaker'].analog_proc, f"amp{entry['buffer']}")] = scale
        return gains

    def update_gains(self, t, force=False):
        """Write the gains that changed since the last call; returns the number of writes."""
        n = 0
        for key, gain in self._target_gains(t).items():
            last = self._written.get(key)
            if force or last is None or abs(last - gain) > MIN_CHANGE or (gain == 0) != (last == 0):
                proc, tag = key
                freefield_core.PROCESSORS.processors[proc].SetTagVal(tag, float(gain))
                self._written[key] = gain
                n += 1
        return n

    # ------------------------------------------------------------ live balance
    def adjust(self, index, delta_db):
        """Turn sound `index` (0-based, order of SOUNDS) up or down by `delta_db`."""
        entry = self.sounds[index]
        new_offset = entry['offset'] + delta_db
        if entry['peak'] * 10 ** (new_offset / 20) >= 1:
            print(f"  {entry['name']} would clip at {entry['level'] + new_offset:g} dB -- not changed.")
            return
        entry['offset'] = new_offset

    def print_levels(self):
        """Current levels -- copy the numbers into SOUNDS."""
        for i, entry in enumerate(self.sounds, start=1):
            print(f"  {i:2d} {entry['name']:8s} {entry['level'] + entry['offset']:5g} dB"
                  f"{'  moving' if entry['moving'] else ''}{'  (muted)' if entry['muted'] else ''}")

    # ------------------------------------------------------------ run
    def run(self):
        """Load, start, move the moving sounds, and handle console commands until 'q'.

        All processor calls stay in this thread (the TDT COM objects belong to it);
        a second thread only reads console lines into a queue.
        """
        self.init_processors()
        self.load()
        self.start()
        help_text = (f'Commands:  <n>  mute/unmute sound n (1-{len(self.sounds)})  |  '
                     '<n> <dB>  e.g. "6 -3" turns sound 6 down by 3 dB  |  '
                     'a all on | m all off | l levels | s stop | p play | q quit')
        print(help_text)
        self.print_levels()
        commands = queue.Queue()

        def read_console():
            while True:
                commands.put(input())

        threading.Thread(target=read_console, daemon=True).start()
        try:
            while True:
                tick = time.monotonic()
                if self._t0 is not None:
                    self.update_gains(tick - self._t0)
                try:
                    line = commands.get(timeout=max(0.0, 1 / UPDATE_RATE - (time.monotonic() - tick)))
                except queue.Empty:
                    continue
                if not self._command(line, help_text):
                    break
        finally:
            self.stop()

    def _command(self, line, help_text):
        """Handle one console line; returns False to quit."""
        parts = line.strip().lower().split()
        if not parts:
            return True
        command = parts[0]
        if command == 'q':
            return False
        elif command == 's':
            self.stop()
        elif command == 'p':
            self.start()
        elif command in ('a', 'm'):
            for entry in self.sounds:
                entry['muted'] = command == 'm'
        elif command == 'l':
            self.print_levels()
        elif command.isdigit() and 1 <= int(command) <= len(self.sounds):
            index = int(command) - 1
            if len(parts) == 1:
                self.sounds[index]['muted'] = not self.sounds[index]['muted']
            else:
                try:
                    self.adjust(index, float(parts[1]))
                except ValueError:
                    print(help_text)
                    return True
            self.print_levels()
        else:
            print(help_text)
        if self._t0 is None:  # stopped: still apply mutes and levels for the next start
            self.update_gains(0.0)
        return True


# %% run
if __name__ == '__main__':
    demo = DomeDemo()
    demo.run()
