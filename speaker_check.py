"""
Speaker check -- play a short noise burst from one dome loudspeaker after the other.

Uses the standard freefield setup: play_buf.rcx on RX81 and RX82, and the dome
speaker table, so it tests exactly the wiring freefield (and the demo) assume.
For each speaker it prints index, RX8, channel, azimuth and elevation; compare
that with where you hear it.

Order: column by column (azimuth left to right), each from bottom to top.

Console, after each speaker:
    Enter        next speaker
    r            repeat
    b            back one
    <number>     jump to speaker index <number> (as in the table)
    a            auto: go through the rest with AUTO_PAUSE s per speaker (Ctrl+C stops)
    q            quit
    any text     note for this speaker (e.g. "heard at 17.5 / 0"); saved at the end
"""
# %% imports
import csv
import logging
import time
from datetime import datetime
from pathlib import Path

import slab
import freefield
# `import freefield` only exposes a copy of PROCESSORS/SPEAKERS (star import),
# so the module itself is needed to see them after initialize().
from freefield import freefield as freefield_core

logging.basicConfig(level=logging.INFO)

# ==================== quick config ====================
LEVEL = 70              # dB (slab, uncalibrated), the burst level
N_BURSTS = 3            # bursts per speaker
BURST = 0.25            # s, burst length
GAP = 0.15              # s, silence between bursts
AUTO_PAUSE = 1.0        # s, extra pause per speaker in auto mode
EQUALIZE = False        # no dome equalization measured at the moment
# ======================================================
OUT_DIR = Path(__file__).resolve().parent


def init():
    rcx = Path(freefield.DIR) / 'data' / 'rcx' / 'play_buf.rcx'
    freefield.initialize('dome', device=[['RX81', 'RX8', str(rcx)], ['RX82', 'RX8', str(rcx)]],
                         zbus=True, sensor_tracking=False)
    rate = freefield_core.PROCESSORS.processors['RX81'].GetSFreq()
    logging.info('play_buf.rcx running at %.0f Hz', rate)
    return rate


def make_stimulus(rate):
    """N_BURSTS pink-noise bursts with gaps, as one sound (easy to localize)."""
    slab.set_default_samplerate(rate)
    burst = slab.Sound.pinknoise(duration=BURST, samplerate=rate).ramp(duration=0.01)
    burst.level = LEVEL
    gap = slab.Sound.silence(duration=GAP, samplerate=rate)
    stim = burst
    for _ in range(N_BURSTS - 1):
        stim = slab.Sound.sequence(stim, gap, burst)
    return stim


def play(stim, speaker):
    freefield.set_signal_and_speaker(signal=stim, speaker=speaker.index, equalize=EQUALIZE)
    freefield.play()
    time.sleep(stim.duration + 0.05)


def describe(speaker):
    return (f'speaker {speaker.index:2d} | {speaker.analog_proc} ch {speaker.analog_channel:2d} | '
            f'az {speaker.azimuth:6.1f}  el {speaker.elevation:6.1f}')


def main():
    rate = init()
    stim = make_stimulus(rate)
    speakers = sorted(freefield_core.SPEAKERS, key=lambda s: (s.azimuth, s.elevation))
    by_index = {s.index: k for k, s in enumerate(speakers)}
    notes = {}
    print(__doc__.split('Console, after each speaker:')[1])
    k, auto = 0, False
    try:
        while 0 <= k < len(speakers):
            spk = speakers[k]
            print(f'[{k + 1:2d}/{len(speakers)}] {describe(spk)}')
            play(stim, spk)
            if auto:
                time.sleep(AUTO_PAUSE)
                k += 1
                continue
            answer = input('  > ').strip()
            if answer == '':
                k += 1
            elif answer.lower() == 'q':
                break
            elif answer.lower() == 'r':
                continue
            elif answer.lower() == 'b':
                k = max(0, k - 1)
            elif answer.lower() == 'a':
                auto = True
                k += 1
            elif answer.isdigit() and int(answer) in by_index:
                k = by_index[int(answer)]
            else:
                notes[spk.index] = answer
                print(f'  noted for speaker {spk.index}: {answer}')
                k += 1
    except KeyboardInterrupt:
        print('\nstopped')
    finally:
        freefield.halt()
    if notes:
        file = OUT_DIR / f'speaker_check_{datetime.now():%Y%m%d_%H%M}.csv'
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'proc', 'channel', 'azimuth', 'elevation', 'note'])
            for spk in speakers:
                if spk.index in notes:
                    writer.writerow([spk.index, spk.analog_proc, spk.analog_channel,
                                     spk.azimuth, spk.elevation, notes[spk.index]])
        print(f'{len(notes)} notes saved to {file}')
        for spk in speakers:
            if spk.index in notes:
                print(f'  {describe(spk)}  ->  {notes[spk.index]}')


# %% run
if __name__ == '__main__':
    main()
