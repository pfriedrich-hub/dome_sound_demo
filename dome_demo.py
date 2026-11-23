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
      sigin{i}         SerialBuf, fixed Size = BUFFER_SIZE, wraps at Size (as in Marc's circuit).
                       The script fills the whole buffer with repetitions of the sound, so it
                       loops at its own length; at the wrap (every BUFFER_SIZE samples, 45 s at
                       24414 Hz) the last repetition is cut short and faded out.
      aSig{i}_Ch{c}    gain of buffer i on channel c = 1..24, smoothed (as in Marc's circuit)
    buffers in STATIC_WIRING (per RX8) -- static sounds
      sigin{i} as above
      amp{i}           gain of buffer i, smoothed; output added to ONE channel, wired in the circuit
    per channel c      sum of everything routed to c -> one DacOut c
                       (one DacOut per channel: TDT does not document what two DacOuts
                       on the same channel do)
    zBusA (level)      SerialBuf AccEnab: high = play, low = stop (as in Marc's circuit)
    zBusB              SerialBuf Rst: a pulse resets every buffer index to 0

A hopping sound (cat, cows) is a static buffer wired to several channels, each through
its own gain amp{i}_Ch{c}; the script opens one of them per call, so every call comes
from one speaker and the next call from another.

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


# %% hopping -- a sound that plays every call (meow, moo, ...) from one speaker of a set and
# jumps to another speaker in the pause before the next call. The circuit feeds its buffer to
# each of these speakers through its own gain (amp{i}_Ch{c}); no panning, no smearing.
class Hop:
    def __init__(self, positions, choose):
        self.positions = positions    # [(azimuth, elevation), ...] -- all on the same RX8
        self.choose = choose          # choose(current index, positions, rng) -> next index


class Spread:
    """A static sound played from several speakers at once (equal power), e.g. a wide surf."""
    def __init__(self, positions):
        self.positions = positions    # [(azimuth, elevation), ...] -- all on the same RX8


def wander(max_jump=40, p_up=0.6, p_drop=0.15):
    """Random jumps to a speaker up to `max_jump` deg away -- upwards with probability `p_up` --
    and now and then (`p_drop`) a drop back to the lowest row."""
    def choose(current, positions, rng):
        vectors = _unit_vectors(*numpy.array(positions, dtype=float).T)
        angle = numpy.degrees(numpy.arccos(numpy.clip(vectors @ vectors[current], -1, 1)))
        elevation = numpy.array([el for _, el in positions])
        others = numpy.arange(len(positions)) != current
        if rng.random() < p_drop:
            lowest = numpy.flatnonzero(others & (elevation == elevation[others].min()))
            return int(rng.choice(lowest))
        near = numpy.flatnonzero(others & (angle <= max_jump))
        up = near[elevation[near] > elevation[current]]
        if len(up) and rng.random() < p_up:
            return int(rng.choice(up))
        return int(rng.choice(near if len(near) else numpy.flatnonzero(others)))
    return choose


def scatter():
    """Each call from a random other speaker of the set."""
    def choose(current, positions, rng):
        return int(rng.choice([k for k in range(len(positions)) if k != current]))
    return choose


# ==================== quick config ====================
# Static sounds: wav file, (azimuth, elevation) of the dome speaker, level in dB.
# Hopping sounds: wav file, a Hop (speakers + jump rule, see above), level in dB.
# Moving sounds: wav file, a path from the functions above, level in dB.
# Optional 4th element, a dict of options:
#   'gap': (min, max)      seconds of silence after each repetition of the sound in its buffer,
#                          drawn at random -- a short call then does not repeat at a fixed rhythm.
#   'snippet': (min, max)  instead of the whole file, play random excerpts of min..max s
#                          (faded in and out), each followed by a 'gap'.
# A file can appear more than once (e.g. at several speakers); every copy gets its own
# random excerpts and pauses.
# The level is the ACTIVE level: the RMS over the parts where the sound is on (100 ms
# windows less than ACTIVE_RANGE dB below the loudest one), not over the whole file --
# the cat is silent ~80 % of its file, so its plain RMS is ~7 dB below its meows.
# The cat is the target to localize: static, and small but above the ocean bed.
# These levels are a starting point -- balance them by ear in the dome with the
# console (e.g. '6 -2'), then copy the values the console prints back here.
SOUND_DIR = Path(__file__).resolve().parent / 'sounds'   # wav paths are relative to this
BIRDS = {'snippet': (2, 5), 'gap': (8, 25)}               # options shared by the five bird copies
MOVE_PERIOD = 90        # s, one round trip of geese and train (one crossing = half of it)
CAT = Hop([(-35, -37.5), (-35, -25), (-35, -12.5), (-35, 0), (-35, 12.5), (-35, 25), (-35, 37.5),
           (-52.5, -12.5), (-52.5, 0), (-17.5, -37.5), (-17.5, -25), (-17.5, -12.5)],
          wander(max_jump=40, p_up=0.6, p_drop=0.15))   # left: climbs, jumps, drops down
COWS = Hop([(0, -25), (17.5, -25), (35, -25),                # the pen, right and below:
            (0, -12.5), (17.5, -12.5), (35, -12.5)], scatter())  # each moo from another cow
SURF = Spread([(0, -50), (0, -37.5), (17.5, -37.5), (35, -37.5)])  # bottom row, centre-right
SOUNDS = [
    ('ocean.wav', SURF, 52),                                             # spread, low right
    ('chimes.wav', (-17.5, 25), 54),
    ('chimes2.wav', (0, 37.5), 52),
    ('chimes3.wav', (17.5, 25), 53),
    ('cat.wav', CAT, 52),                                                # target
    ('cows.wav', COWS, 62),
    ('motor.wav', (52.5, -12.5), 54),                                    # right, below the horizon
    # birds: short random excerpts at random times from five places up high, so they
    # pop up now here, now there (each copy draws its own excerpts and pauses)
    ('birds.wav', (-52.5, 25), 56, BIRDS),
    ('birds.wav', (-52.5, 12.5), 56, BIRDS),
    ('birds.wav', (0, 50), 56, BIRDS),
    ('birds.wav', (35, 37.5), 56, BIRDS),
    ('birds.wav', (52.5, 25), 56, BIRDS),
    # moving: geese fly over left -> right, the train passes slowly in the background
    # right -> left, along the horizon. Same period, so they always move in opposite
    # directions (they meet in the middle, then swap sides).
    ('geese.wav', ping_pong((-52.5, 30), (52.5, 30), period=MOVE_PERIOD), 57,
     {'gap': (1, 4)}),                                                   # 1-4 s silence between calls
    ('train_s.wav', ping_pong((52.5, 0), (-52.5, 0), period=MOVE_PERIOD), 50),
    # ('train.wav', ...) -- 184 s: a buffer that long does not fit 8 times into the RX8
    #                       memory; train_s.wav is Marc's trimmed version of it.
]
WIDTH = 30              # deg, FWHM of the Gaussian that pans a moving sound (Marc's was ~90)
UPDATE_RATE = 10        # Hz, how often the gains of moving sounds are rewritten
MASTER_GAIN = 0         # dB added to every level -- turns the whole scene up or down
ACTIVE_RANGE = 30       # dB below the loudest 100 ms window that still counts as 'on'
GAP_SEED = 1            # random gaps/excerpts are the same on every run (and on both RX8s)
SNIPPET_FADE = 0.2      # s, fade in/out of each random excerpt
HOP_PAUSE = 0.4         # s, shorter pauses do not split a call (a hopping sound only jumps in longer ones)
LOOP_RAMP = 0.005       # s, raised-cosine ramp at both ends of each sound, so the wrap does not click
# ======================================================

# Fixed in the circuit -- these must match it (see the docstring):
SAMPLERATE = 24414      # the wavs are 22.05 kHz (content up to 11 kHz); 24414 Hz keeps it
                        # (the DAC filter may roll off slightly at the very top) and costs
                        # half the DSP load of 48828 Hz
# The rate is the one stored in the .rcx (RPvdsEx device setup); init_processors checks it.
N_MOVING = 2            # buffers 1..2 on both RX8s: sigin, len, aSig{i}_Ch1..24
                        # (3 x 24 gain cells do not fit on two 2-DSP RX8s: >256 components/DSP)
N_CHANNELS = 24         # aSig{i}_Ch1..24
STATIC_WIRING = {       # static buffer -> the DAC channel(s) it is wired to in that RX8's circuit
    # one channel: gain tag amp{i};  several channels (hopping sound): one gain tag amp{i}_Ch{c} each
    'RX81': {3: 5,                    # birds (0, 50)
             4: 11, 5: 4, 6: 20,      # chimes, chimes2, chimes3
             7: [15, 7, 21, 16, 8, 24],  # cows: az 0 / 17.5 / 35 at el -25 and -12.5
             8: [13, 14, 6, 22]},     # ocean: az 0 el -50 / -37.5, az 17.5 and 35 el -37.5
    'RX82': {3: [17, 24, 2, 1, 4, 3, 23,  # cat: az -35, el -37.5 ... 37.5
                 21, 20,                  #      az -52.5, el -12.5 / 0
                 5, 8, 7],                #      az -17.5, el -37.5 / -25 / -12.5
             4: 11,                   # motor (52.5, -12.5)
             5: 22, 6: 13, 7: 19, 8: 16},  # birds (-52.5, 25), (35, 37.5), (-52.5, 12.5), (52.5, 25)
}
BUFFER_SIZE = 1100000   # SerialBuf Size in the circuits -- must be EXACTLY that value: every buffer is
                        # filled to it and wraps there (45 s at 24414 Hz). The longest sound, cows.wav
                        # (42 s), needs 1.03 M at 24414 Hz, so every sound fits at least once.
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


def _fill_buffer(data, n_samples=BUFFER_SIZE, gap=None, snippet=None, seed=GAP_SEED):
    """Repeat `data` until it is `n_samples` long; fade out the cut-off last repetition.

    The circuit's buffers have a fixed Size and wrap there, so a buffer that holds
    the sound back to back loops seamlessly except at that one wrap point.
    gap=(min, max): a random min..max s of silence follows each repetition.
    snippet=(min, max): each repetition is a random min..max s excerpt of `data`.
    """
    if snippet is not None:
        rng = numpy.random.default_rng(seed)
        n_fade = int(SNIPPET_FADE * SAMPLERATE)
        fade = numpy.sin(numpy.linspace(0, numpy.pi / 2, n_fade)) ** 2
        lead = numpy.zeros(int(rng.uniform(0, (gap or (0, 0))[1]) * SAMPLERATE))  # copies start apart
        pieces, n = [lead], len(lead)
        while n < n_samples:
            length = min(len(data), int(rng.uniform(*snippet) * SAMPLERATE))
            start = rng.integers(0, len(data) - length + 1)
            piece = data[start:start + length].copy()
            piece[:n_fade] *= fade
            piece[-n_fade:] *= fade[::-1]
            silence = numpy.zeros(int(rng.uniform(*(gap or (0, 0))) * SAMPLERATE))
            pieces += [piece, silence]
            n += len(piece) + len(silence)
        filled = numpy.concatenate(pieces)[:n_samples]
    elif gap is None:
        filled = numpy.tile(data, int(numpy.ceil(n_samples / len(data))))[:n_samples]
    else:
        rng = numpy.random.default_rng(seed)
        pieces, n = [], 0
        while n < n_samples:
            silence = numpy.zeros(int(rng.uniform(*gap) * SAMPLERATE))
            pieces += [data, silence]
            n += len(data) + len(silence)
        filled = numpy.concatenate(pieces)[:n_samples]
    n_ramp = max(1, int(LOOP_RAMP * SAMPLERATE))  # fade out whatever is cut at the wrap
    filled[-n_ramp:] *= numpy.cos(numpy.linspace(0, numpy.pi / 2, n_ramp)) ** 2
    return filled


def _events(data, window=0.05):
    """(starts, ends) in samples of the calls in `data`: stretches less than ACTIVE_RANGE dB
    below the loudest 50 ms window, with pauses shorter than HOP_PAUSE bridged."""
    n_win = max(1, int(window * SAMPLERATE))
    frames = data[:len(data) // n_win * n_win].reshape(-1, n_win)
    power = (frames ** 2).mean(axis=1)
    on = numpy.concatenate([[0], (power >= power.max() * 10 ** (-ACTIVE_RANGE / 10)).astype(int), [0]])
    starts, ends = numpy.flatnonzero(numpy.diff(on) == 1), numpy.flatnonzero(numpy.diff(on) == -1)
    keep_s, keep_e = [starts[0]], [ends[0]]
    for a, b in zip(starts[1:], ends[1:]):
        if (a - keep_e[-1]) * n_win < HOP_PAUSE * SAMPLERATE:
            keep_e[-1] = b
        else:
            keep_s.append(a); keep_e.append(b)
    return numpy.array(keep_s) * n_win, numpy.array(keep_e) * n_win


# %% demo
class DomeDemo:
    """Load the sounds in SOUNDS onto the RX8s, loop them, and move the moving ones."""

    def __init__(self, sounds=SOUNDS):
        self.table = _dome_table()
        self.speaker_vectors = _unit_vectors(numpy.array([s.azimuth for s in self.table]),
                                             numpy.array([s.elevation for s in self.table]))
        self.sounds = []  # one dict per sound, in the order of SOUNDS
        for n_sound, (file, where, level, *options) in enumerate(sounds):
            options = options[0] if options else {}
            entry = {'name': Path(file).stem, 'level': level, 'offset': 0.0, 'muted': False,
                     'sound': _load_sound(file, level)}
            if [f for f, *_ in sounds].count(file) > 1 and not callable(where):
                entry['name'] += f'@{where[0]:g},{where[1]:g}'   # several copies: tell them apart
            if callable(where):
                entry.update(moving=True, hopping=False, spread=False, path=where, procs=PROCS)
            elif isinstance(where, (Hop, Spread)):
                speakers = [_find_speaker(self.table, *pos) for pos in where.positions]
                procs = {s.analog_proc for s in speakers}
                if len(procs) > 1:
                    raise ValueError(f"{entry['name']}: hopping speakers on {procs} -- they must all be "
                                     'on one RX8 (its buffer reaches only that RX8).')
                entry.update(moving=False, hopping=isinstance(where, Hop), spread=isinstance(where, Spread),
                             hop=where, speakers=speakers, procs=(speakers[0].analog_proc,),
                             rng=numpy.random.default_rng(), current=0, event=0)
            else:
                speaker = _find_speaker(self.table, *where)
                entry.update(moving=False, hopping=False, spread=False, speaker=speaker,
                             procs=(speaker.analog_proc,))
            data = numpy.asarray(entry['sound'].data, dtype=float).flatten()
            if len(data) > BUFFER_SIZE:
                raise ValueError(f"{entry['name']} is {len(data)} samples, the buffers hold "
                                 f'{BUFFER_SIZE}. Raise BUFFER_SIZE and the SerialBuf Size.')
            entry['n_samples'] = len(data)
            entry['data'] = _fill_buffer(data, gap=options.get('gap'), snippet=options.get('snippet'),
                                         seed=GAP_SEED + n_sound)
            entry['peak'] = numpy.abs(entry['data']).max()
            if entry['hopping']:
                entry['starts'], entry['ends'] = _events(entry['data'])
                entry['current'] = int(entry['rng'].integers(len(entry['speakers'])))
            if entry['peak'] >= 1:  # +-1 is +-10 V on the RX8 DAC
                logging.warning('%s peaks at %.2f (>= 1): lower its level or it will clip.',
                                entry['name'], entry['peak'])
            self.sounds.append(entry)
        self._check_positions()
        self._assign_buffers()
        self._written = {}      # (proc, buffer, channel) -> last gain written
        self._t0 = None
        self._fs = SAMPLERATE   # the RX8's exact rate, read in init_processors
        self._verbose = False   # 'v' in the console: print every jump of a hopping sound
        for proc in PROCS:
            n = N_MOVING + len(STATIC_WIRING[proc])
            logging.info('%s: %d buffers x %d samples = %.0f MB at %d Hz.',
                         proc, n, BUFFER_SIZE, n * BUFFER_SIZE * 4 / 1e6, SAMPLERATE)

    def _check_positions(self):
        multi = [s for s in self.sounds if s['hopping'] or s['spread']]
        used = [s['speaker'].index for s in self.sounds if not s['moving'] and s not in multi]
        used += [spk.index for s in multi for spk in s['speakers']]
        if len(used) != len(set(used)):
            raise ValueError('Two static/hopping sounds share a speaker.')

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
            if entry['hopping'] or entry['spread']:
                proc = entry['procs'][0]
                channels = {spk.analog_channel for spk in entry['speakers']}
                match = [i for i, c in STATIC_WIRING[proc].items()
                         if isinstance(c, list) and set(c) == channels]
                if not match:
                    raise ValueError(f"{entry['name']}: no buffer on {proc} is wired to exactly the "
                                     f'channels {sorted(channels)}. Wired: {STATIC_WIRING[proc]}.')
                entry['buffer'] = match[0]
                continue
            speaker = entry['speaker']
            wired = {c: i for i, c in STATIC_WIRING[speaker.analog_proc].items() if not isinstance(c, list)}
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
            self._fs = rate

    def load(self):
        """Write all buffers, lengths and initial gains. Call while the demo is stopped."""
        self.stop()
        for proc in PROCS:
            for entry in self.sounds:
                if proc in entry['procs']:
                    i = entry['buffer']
                    t_write = time.monotonic()
                    freefield.write(f'sigin{i}', entry['data'], proc)
                    logging.info('%s: %-15s -> sigin%d (%d samples, %.1f s)', proc, entry['name'], i,
                                 len(entry['data']), time.monotonic() - t_write)
        self._written.clear()
        self.update_gains(t=0.0, force=True)
        for entry in self.sounds:
            if entry['moving']:
                where = 'moving'
            elif entry['hopping']:
                where = f"hopping over {len(entry['speakers'])} speakers, {len(entry['starts'])} calls"
            elif entry['spread']:
                where = f"spread over {len(entry['speakers'])} speakers"
            else:
                where = (f"speaker {entry['speaker'].index:2d} (az {entry['speaker'].azimuth:g}, "
                         f"el {entry['speaker'].elevation:g})")
            logging.info('%-15s buffer %d on %s | %6.2f s | %d dB | %s', entry['name'],
                         entry['buffer'], '+'.join(entry['procs']),
                         entry['n_samples'] / SAMPLERATE, entry['level'], where)

    @staticmethod
    def _zbus(trigger, kind):
        """zBus trigger to both RX8s. kind: 0 = pulse, 1 = high, 2 = low (as in Marc's script)."""
        zbus = freefield_core.PROCESSORS._zbus
        (zbus.zBusTrigA if trigger == 'A' else zbus.zBusTrigB)(0, kind, 10)

    def start(self):
        """All buffers back to sample 0, then play -- on both RX8s in the same sample."""
        self._zbus('A', 2)  # AccEnab low
        self._zbus('B', 0)  # Rst pulse: indices to 0
        self._zbus('A', 1)  # AccEnab high: play
        self._t0 = time.monotonic()

    def stop(self):
        self._zbus('A', 2)
        self._t0 = None

    # ------------------------------------------------------------ gains
    def _target_gains(self, t):
        """{(proc, tag): gain} for every gain tag in the circuits, at scene time t."""
        gains = {(proc, f'aSig{i}_Ch{c}'): 0.0 for proc in PROCS
                 for i in range(1, N_MOVING + 1) for c in range(1, N_CHANNELS + 1)}
        for proc in PROCS:
            for i, channels in STATIC_WIRING[proc].items():
                if isinstance(channels, list):
                    gains.update({(proc, f'amp{i}_Ch{c}'): 0.0 for c in channels})
                else:
                    gains[(proc, f'amp{i}')] = 0.0
        for entry in self.sounds:
            scale = 0.0 if entry['muted'] else 10 ** (entry['offset'] / 20)
            if entry['hopping']:
                speaker = entry['speakers'][entry['current']]
                gains[(speaker.analog_proc, f"amp{entry['buffer']}_Ch{speaker.analog_channel}")] = scale
            elif entry['spread']:   # equal power over all its speakers
                for speaker in entry['speakers']:
                    gains[(speaker.analog_proc, f"amp{entry['buffer']}_Ch{speaker.analog_channel}")] = \
                        scale / math.sqrt(len(entry['speakers']))
            elif entry['moving']:
                weights = pan_weights(*entry['path'](t), self.speaker_vectors) * scale
                for speaker, w in zip(self.table, weights):
                    gains[(speaker.analog_proc, f"aSig{entry['buffer']}_Ch{speaker.analog_channel}")] = w
            else:
                gains[(entry['speaker'].analog_proc, f"amp{entry['buffer']}")] = scale
        return gains

    def _update_hops(self, t):
        """In the pause before each call of a hopping sound, choose its next speaker.

        The position in the buffer follows from the time since start() at the RX8's
        exact rate (all buffers start in the same sample and wrap at BUFFER_SIZE).
        """
        pos = int(t * self._fs) % BUFFER_SIZE
        for entry in self.sounds:
            if not entry['hopping']:
                continue
            starts, ends = entry['starts'], entry['ends']
            k = int(numpy.searchsorted(starts, pos, side='right')) - 1   # last call that started
            if k >= 0 and pos < ends[k]:
                continue                                  # inside a call: never jump
            upcoming = (k + 1) % len(starts)
            if entry['event'] != upcoming:
                entry['event'] = upcoming
                entry['current'] = entry['hop'].choose(entry['current'], entry['hop'].positions,
                                                       entry['rng'])
                if self._verbose and not entry['muted']:
                    az, el = entry['hop'].positions[entry['current']]
                    print(f"  {entry['name']} -> az {az:g}, el {el:g}")

    def update_gains(self, t, force=False):
        """Write the gains that changed since the last call; returns the number of writes."""
        if self._t0 is not None:
            self._update_hops(t)
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
        old = entry['level'] + entry['offset']
        entry['offset'] = new_offset
        print(f"  {entry['name']}: {old:g} -> {entry['level'] + new_offset:g} dB"
              f"{'  (still muted)' if entry['muted'] else ''}")

    def print_levels(self):
        """Current levels -- copy the numbers into SOUNDS."""
        for i, entry in enumerate(self.sounds, start=1):
            print(f"  {i:2d} {entry['name']:15s} {entry['level'] + entry['offset']:5g} dB"
                  f"{'  moving' if entry['moving'] else ''}{'  hopping' if entry['hopping'] else ''}"
                  f"{'  spread' if entry['spread'] else ''}"
                  f"{'  (muted)' if entry['muted'] else ''}")

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
                     'a all on | m all off | l levels | v hop printout | s stop | p play | q quit')
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
        """Handle one console line, print what changed; returns False to quit."""
        parts = line.strip().lower().split()
        if not parts:
            return True
        command = parts[0]
        if command == 'q':
            print('  quit')
            return False
        elif command == 's':
            self.stop()
            print('  stopped')
        elif command == 'p':
            self.start()
            print('  playing (all buffers from the start)')
        elif command in ('a', 'm'):
            for entry in self.sounds:
                entry['muted'] = command == 'm'
            print(f"  all {len(self.sounds)} sounds {'muted' if command == 'm' else 'on'}")
        elif command == 'l':
            self.print_levels()
        elif command == 'v':
            self._verbose = not self._verbose
            print(f"  hop printout {'on' if self._verbose else 'off'}")
        elif command.isdigit() and 1 <= int(command) <= len(self.sounds):
            index = int(command) - 1
            entry = self.sounds[index]
            if len(parts) == 1:
                entry['muted'] = not entry['muted']
                state = 'muted' if entry['muted'] else f"on ({entry['level'] + entry['offset']:g} dB)"
                print(f"  {entry['name']}: {state}")
            else:
                try:
                    self.adjust(index, float(parts[1]))
                except ValueError:
                    print(help_text)
                    return True
        else:
            print(help_text)
        if self._t0 is None:  # stopped: still apply mutes and levels for the next start
            self.update_gains(0.0)
        return True


# %% run
if __name__ == '__main__':
    demo = DomeDemo()
    demo.run()
